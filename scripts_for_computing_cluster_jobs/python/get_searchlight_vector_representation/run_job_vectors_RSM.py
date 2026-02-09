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



all_4_names = ['sameEv-sameSchema',
                'otherEv-sameSchema',
                'sameEv-otherSchema',
                'otherEv-otherSchema']

# mark where each event starts and ends
event_1_start = 17
event_2_start = 23
event_3_start = 34
event_4_start = 50
event_5_start = 66

def run_jobs(input_dir, json_path, testing = False,  
            output_dir = "/scratch/network/rk1593/", 
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
            go_from_480searchlight_files_representing_fingerprintPlot_to_tvalue_vector(chunk, 
                                         testing, 
                                         output_dir,
                                        input_dir,
                                        )
            # new_process = Process(target = go_from_480searchlight_files_representing_fingerprintPlot_to_tvalue_vector, 
            #                 args = (   chunk, 
            #                              testing, 
            #                              output_dir, 
            #                             input_dir,
            #                             ))
            # new_process.start()
            # processes_list.append(new_process)
        for p in processes_list:
            p.join()

# requires: searchlight_to_files_tuples for all searchlights, subset_list_of_searchlights are a list of the searchlights we want to include for this job
# outputs a csv file for each searchlight
def go_from_480searchlight_files_representing_fingerprintPlot_to_tvalue_vector(subset_list_of_searchlights, 
            testing = False, 
            output_dir = "",
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
            np.savetxt(output_dir +  "/searchlights_collapseTR_just_means/" + light_id + "_NOT480.csv", np.array([1,2,3]), delimiter=",")
            return
        # don't reprocess a searchlight again if we already did create
        # tvalue vector for it in the past
        save_path = output_dir +  "/searchlights_collapseTR_just_means/" + light_id + ".csv"
        if os.path.exists(save_path):
            print("Path Exists, Do Not Reprocess")
            continue
        template2_count = 0
        template3_count = 0
        template4_count = 0
        template_to_pid_to_cond_to_matrices = {} # create this dict for later usage when creating fingerprint plots
        template_to_pid_to_cond_to_event_to_mean = {}
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
                mean_all_events = np.nanmean(new_arr, axis = 0).tolist()
                mean_event2 = np.nanmean(mean_all_events[event_2_start:event_3_start])
                mean_event3 = np.nanmean(mean_all_events[event_3_start:event_4_start])
                mean_event4 = np.nanmean(mean_all_events[event_4_start:event_5_start])
                new_arr = new_arr.tolist()
            # error check if the mean list has an nan in it which means that all weddings in one tr had nan
            if np.isnan(mean_event2) or np.isnan(mean_event3) or np.isnan(mean_event4):
                print("Error: mean is np.nan", light_id)
                # if isnan has all Falses and so everything is 0
                return
            if str(template_id) not in template_to_pid_to_cond_to_matrices:
                template_to_pid_to_cond_to_matrices[str(template_id)] = {}
                template_to_pid_to_cond_to_event_to_mean[template_id] = {}
            if pid not in template_to_pid_to_cond_to_matrices[str(template_id)]:
                template_to_pid_to_cond_to_matrices[str(template_id)][pid] = {}
                template_to_pid_to_cond_to_event_to_mean[template_id][pid] = {}
            template_to_pid_to_cond_to_matrices[str(template_id)][pid][cond] = new_arr
            template_to_pid_to_cond_to_event_to_mean[template_id][pid][cond] = {}
            template_to_pid_to_cond_to_event_to_mean[template_id][pid][cond][2] = mean_event2
            template_to_pid_to_cond_to_event_to_mean[template_id][pid][cond][3] = mean_event3
            template_to_pid_to_cond_to_event_to_mean[template_id][pid][cond][4] = mean_event4

        # check that we got even number for each template for error check
        if template2_count != 160 or template3_count != 160 or template4_count != 160:
            print("Error!")
            print("tar_file_name: ", tar_file_name)
            print("template2_count: ", template2_count)
            print("template3_count: ", template3_count)
            print("template4_count: ", template4_count)
            return
        template_to_event_to_cond_mean = {}
        for template_id in template_to_pid_to_cond_to_event_to_mean:
            template_to_event_to_cond_mean[template_id] = {}
            for event_id in [2,3,4]:
                template_to_event_to_cond_mean[template_id][event_id] = {}
                for cond in all_4_names:
                    list_of_points_in_violin = []
                    for pid in template_to_pid_to_cond_to_event_to_mean[template_id]:
                        one_point = \
                        template_to_pid_to_cond_to_event_to_mean[template_id][pid][cond][event_id]
                        list_of_points_in_violin.append(one_point)
                    assert(len(list_of_points_in_violin) == 40)
                    template_to_event_to_cond_mean[template_id][event_id][cond] = np.mean(list_of_points_in_violin)
        features = []
        for template_id in [2,3,4]:
            for cond in all_4_names:
                for event_id in [2,3,4]:
                    stat = template_to_event_to_cond_mean[template_id][event_id][cond]
                    features.append(stat)
        assert(len(features) == 36)
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
input_dir = "/scratch/gpfs/rk1593/tar_by_searchlight/480_files_each_searchlight/" # here we have stored a list of 480 files in a tar file for each searchlight
output_dir = "/scratch/gpfs/rk1593/clustering_output/"  # output dhere on della
json_file_name = "jobs_info_dict_manual_jupyter_without_tuples.json"
testing = False
num_chunks = 31
job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
print("job_id_in: ", job_id_in)
if bash_it:
    run_jobs(input_dir, json_path = output_dir + json_file_name, 
                testing = testing, 
            output_dir = output_dir,
            job_id_target= job_id_in,
            num_chunks= num_chunks)
