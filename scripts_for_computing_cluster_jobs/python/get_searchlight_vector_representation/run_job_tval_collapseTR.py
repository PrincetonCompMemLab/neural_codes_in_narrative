import os
import tarfile
import pdb
import csv
import io
import numpy as np
from statistics import stdev
import json
import orjson
import math
from multiprocessing import Process



all_compare_names = ['sameEv-sameSchema_sameEv-otherSchema',
                        'sameEv-sameSchema_otherEv-sameSchema',
                        'sameEv-sameSchema_otherEv-otherSchema',
                        'sameEv-otherSchema_otherEv-sameSchema',
                        'sameEv-otherSchema_otherEv-otherSchema',
                        'otherEv-sameSchema_otherEv-otherSchema']

# mark where each event starts and ends
event_1_start = 17
event_2_start = 23
event_3_start = 34
event_4_start = 50
event_5_start = 66

def run_jobs(input_dir, json_path, testing = False,  
            output_dir = "/scratch/network/rk1593/", 
            only_events2_3_and_4 = True,
            job_id_target = None, num_chunks = 31):
    """
    run a job which is a a particular set of searchlights that we want
    to turn into a tvalue vector for the purpose of clustering
    """
    # open the jobs info dict which is a dictionary containing a mapping
    # from job id (i.e. a number from 0 to 200) to the list of searchlights
    # to process in that job
    f = open(json_path,)
    jobs_info_dict = json.load(f)
    for job_id in range(jobs_info_dict["num_jobs_actual"]):
        # this is how we only process this one job of interest
        if job_id != job_id_target:
            continue
        print("job: ", job_id)
        this_job_searchlights = jobs_info_dict["job_id_to_searchlight_subset"][str(job_id)]
        # break this job into 32 chunks for parallel processing
        chunk_size = math.floor(len(this_job_searchlights) / num_chunks)
        chunks_of_searchlights = divide_chunks(this_job_searchlights, chunk_size = chunk_size)
        processes_list = []
        for index,chunk in enumerate(chunks_of_searchlights):
            print("process index: ", index)
            new_process = Process(target = go_from_480searchlight_files_representing_fingerprintPlot_to_tvalue_vector, 
                            args = (   chunk, 
                                         testing, 
                                         output_dir, 
                                        only_events2_3_and_4,
                                        input_dir,
                                        ))
            new_process.start()
            processes_list.append(new_process)
        for p in processes_list:
            p.join()

# requires: searchlight_to_files_tuples for all searchlights, subset_list_of_searchlights are a list of the searchlights we want to include for this job
# outputs a csv file for each searchlight
def go_from_480searchlight_files_representing_fingerprintPlot_to_tvalue_vector(subset_list_of_searchlights, 
            testing = False, 
            output_dir = "", 
             only_events2_3_and_4 = True,
              in_dir = ""):
    os.chdir(in_dir)
    for counter_light, light_id in enumerate(subset_list_of_searchlights):
        tar_file_name = light_id + "_new.tar.gz"
        # check that we have a tar file for this searchlight
        if not os.path.exists(in_dir + tar_file_name):
            print("__________Light not on della yet!_________")
            continue
        tar = tarfile.open(tar_file_name)
        # get the 480 files and make sure we got the right number
        files_this_light = [x.name for x in tar.getmembers()]
        num_files_this_light = len(files_this_light)
        if num_files_this_light != 480:
            print("Error: searchlight " + light_id + " has " + str(num_files_this_light) + " files." )
            np.savetxt(output_dir +  "/searchlights_tval/" + light_id + "_NOT480.csv", np.array([1,2,3]), delimiter=",")
            return
        # don't reprocess a searchlight again if we already did create
        # tvalue vector for it in the past
        save_path = output_dir +  "/searchlights_tval/" + light_id + ".csv"
        if os.path.exists(save_path):
            print("Path Exists, Do Not Reprocess")
            continue
        template2_count = 0
        template3_count = 0
        template4_count = 0
        template_to_pid_to_cond_to_matrices = {} # create this dict for later usage when creating fingerprint plots
        template_to_pid_to_cond_to_lists = {}
        i = 0
        # step 1 is to get dict mapping the template (2,3 or 4) to pid (N = 40) to cond (condition) where condition is synonymous
        # with "other-Event-same-Schema" and so on
        for file_name in files_this_light:
            splitted_file_name = file_name.split("_")
            # get the participant id
            pid = splitted_file_name[1]
            cond = splitted_file_name[3] + "-" + splitted_file_name[4]
            template_id = int(file_name[-5])
            if file_name[-5] == "2":
                template2_count += 1 
            elif file_name[-5] == "3":
                template3_count += 1 
            elif file_name[-5] == "4":
                template4_count += 1 
            else:
                print("Error: file_name template retrieval error.")
                print(file_name)
                print(tar_file_name)
                return
            # change the directory to this in_dir to be able to open the tar file
            os.chdir(in_dir)
            i += 1
            in_text = tar.extractfile(file_name).read()
            csv_file = io.StringIO(in_text.decode('ascii'))
            random_replacer_for_nothing = str(3e200) # this is set to be something super big (greater than 1), so that 
            # when we replace all numbers outside [-1,1] with NaN then these lines where no data is recorded become NaN
            # cleaning_mean_start = time.time()
            csv_lines = [[y.replace(" ", "") for y in x] for x in csv.reader(csv_file)]
            if testing:
                csv_lines[0][0] = "-1000"
            # replace any spots where there is no data collected with the random replacer (more explained above why I did this)
            csv_lines = [[y if y != "" else random_replacer_for_nothing for y in x] for x in csv_lines]
            try:
                new_arr = np.array(csv_lines).astype("float")
                # replace the empty due to the random replacer and other anomaly large numbers outside of [-1,1] (the range for correlation) found in raw data
                new_arr[abs(new_arr) > 1] = np.nan
                # now we edit the new_arr so that 0-indexed row 9, columns 47 through 73 become nan for subject 102
                if pid == "sub-102": # sub-102 had some issues in this area of the FMRI
                    new_arr[9,47:74] = np.nan
            except ValueError:
                pdb.set_trace()
            # if in testing mode we need to crop out the first column and the first row
            # since these files were different 
            if testing:
                mean_list = np.nanmean(new_arr[1:13, 1:75], axis = 0).tolist()
                new_arr = new_arr[1:13, 1:75].tolist()
            else:
                mean_list = np.nanmean(new_arr, axis = 0).tolist()
                new_arr = new_arr.tolist()
            # error check if the mean list has an nan in it which means that all weddings in one tr had nan
            if np.count_nonzero(np.isnan(mean_list)) != 0:
                print("Error: mean list has np.nan")
                # if isnan has all Falses and so everything is 0
                return 
            if str(template_id) not in template_to_pid_to_cond_to_matrices:
                template_to_pid_to_cond_to_matrices[str(template_id)] = {}
                template_to_pid_to_cond_to_lists[template_id] = {}
            if pid not in template_to_pid_to_cond_to_matrices[str(template_id)]:
                template_to_pid_to_cond_to_matrices[str(template_id)][pid] = {}
                template_to_pid_to_cond_to_lists[template_id][pid] = {}
            template_to_pid_to_cond_to_matrices[str(template_id)][pid][cond] = new_arr
            template_to_pid_to_cond_to_lists[template_id][pid][cond] = mean_list
        with open(output_dir + "searchlights_matrices_orjson/" + light_id, "wb") as f:
            f.write(orjson.dumps(template_to_pid_to_cond_to_matrices))
        # check that we got even number for each template for error check
        if template2_count != 160 or template3_count != 160 or template4_count != 160:
            print("Error!")
            print("tar_file_name: ", tar_file_name)
            print("template2_count: ", template2_count)
            print("template3_count: ", template3_count)
            print("template4_count: ", template4_count)
            return

        # step 2 is to get the dict mapping the template to the comparison
        # where there are 6 differen comparisons between the 4 conditions, to pid to 
        # a list of differences between one condition and another in the comparison
        template_to_compare_to_pid_to_tr_diffs = {}
        for template_id in template_to_pid_to_cond_to_lists:
            if template_id not in template_to_compare_to_pid_to_tr_diffs:
                template_to_compare_to_pid_to_tr_diffs[template_id] = {}
            for pid in template_to_pid_to_cond_to_lists[template_id]:
                for comparison_name in all_compare_names:
                    compare1_name,compare2_name = comparison_name.split("_")    
                    if comparison_name not in template_to_compare_to_pid_to_tr_diffs[template_id]:
                        template_to_compare_to_pid_to_tr_diffs[template_id][comparison_name] = {}
                    compare1_list =  template_to_pid_to_cond_to_lists[template_id][pid][compare1_name]
                    compare2_list = template_to_pid_to_cond_to_lists[template_id][pid][compare2_name]
                    if len(compare1_list) != len(compare2_list):
                        print("Error: for the same template and light, two paths have different number of tr's")
                        return
                    tr_differences = [(compare1_list[i] - compare2_list[i]) for i in range(len(compare1_list))]
                    if only_events2_3_and_4:
                        tr_differences = tr_differences[event_2_start:event_5_start]
                    template_to_compare_to_pid_to_tr_diffs[template_id][comparison_name][pid] = tr_differences

        # step 3: is to get the across subject tvalue for each comparison
        # at each TR for each template and comparison
        template_to_compare_to_trTstats = {}
        for template_id in template_to_compare_to_pid_to_tr_diffs:
            for compare_name in template_to_compare_to_pid_to_tr_diffs[template_id]:
                # check that all participants have the same length of tr_diffs
                # while also getting list of diffs at each tr
                length_tr_list = []
                tr_num_to_list_of_diffs = {}
                for pid in template_to_compare_to_pid_to_tr_diffs[template_id][compare_name]:
                    tr_diffs = template_to_compare_to_pid_to_tr_diffs[template_id][compare_name][pid]
                    for index,diff in enumerate(tr_diffs):
                        if index not in tr_num_to_list_of_diffs:
                            tr_num_to_list_of_diffs[index] = []
                        tr_num_to_list_of_diffs[index].append(diff)
                    length_tr_list.append(len(tr_diffs))
                length_tr_list = np.array(length_tr_list)
                if not np.all(length_tr_list[0] == length_tr_list):
                    pdb.set_trace()
                    print("Error: not all the same length_tr_lists")
                    return
                # now go through each tr index, and get a ttest
                # make sure that we have 40 participants in each tr num
                # and make sure we don't have nan
                for tr_num in tr_num_to_list_of_diffs:
                    if len(tr_num_to_list_of_diffs[tr_num]) != 40 or np.isnan(sum(tr_num_to_list_of_diffs[tr_num])):
                        print(tr_num_to_list_of_diffs[tr_num])
                        pdb.set_trace()
                        print("Error: missing participant or nan")
                        return
               
                tr_Tstats = [get_t_stat_of_list(tr_num_to_list_of_diffs[tr_num]) for tr_num in range(0,len(tr_num_to_list_of_diffs.keys()))]
                if template_id not in template_to_compare_to_trTstats:
                    template_to_compare_to_trTstats[template_id] = {}
                template_to_compare_to_trTstats[template_id][compare_name] = tr_Tstats
                # for each searchlight get a feature list
        # step 4: compile those tstats into one vector which is our final
        # tvalue vector representation that we will cluster
        features = []
        for template_id in [2,3,4]:
            for compare_name in all_compare_names:
                for tr_stat in template_to_compare_to_trTstats[template_id][compare_name]:
                    features.append(tr_stat)
        print("len(features): ", len(features))
        # output it!
        features_np = np.array(features)
        np.savetxt(save_path, features_np, delimiter=",")

def divide_chunks(l, chunk_size):
    """
    take in a list and chunk size and cut this list up into chunks
    """
    # looping till length l
    for i in range(0, len(l), chunk_size): 
        yield l[i:i + chunk_size]

def get_t_stat_of_list(list):
    """
    take in a list of size 40 for each particpant for a particular tr and comparison
    and output the tvalue
    """
    n = len(list)
    mean = (sum(list) / n)
    sd = stdev(list)
    sem = (sd / math.sqrt(n))
    t_stat = (mean / sem)
    return t_stat

# DRIVER #
bash_it = True
input_dir = "/scratch/gpfs/rk1593/tar_by_searchlight/tar_by_searchlight/" # here we have stored a list of 480 files in a tar file for each searchlight
output_dir = "/scratch/gpfs/rk1593/clustering_output/"  # output dhere on della
json_file_name = "jobs_info_dict_manual_jupyter_without_tuples.json"
testing = False
num_chunks = 31
only_events2_3_and_4 = True # we only want to look at event 2, 3 and 4
job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
print("job_id_in: ", job_id_in)
if bash_it:
    run_jobs(input_dir, json_path = output_dir + json_file_name, 
                testing = testing, 
            output_dir = output_dir,  only_events2_3_and_4 = only_events2_3_and_4,
            job_id_target= job_id_in,
            num_chunks= num_chunks)
