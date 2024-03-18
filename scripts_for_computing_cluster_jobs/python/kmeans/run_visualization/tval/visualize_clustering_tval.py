import msgpack
import os
import tarfile
import pdb
import csv
import io
import numpy as np
import json
import math
import pandas as pd
from multiprocessing import Process
from multiprocessing import Manager
import numpy as np
import os


def get_template_to_pid_to_cond_to_matrix(light_id, in_dir = "/scratch/gpfs/rk1593/clustering_output/"):
    template_to_pid_to_cond_to_matrices = {}
    with open(in_dir + "searchlights_matrices_msgpack/" + light_id, "rb") as json_file:
        template_to_pid_to_cond_to_matrices = msgpack.load(json_file, strict_map_key = False) 
    return template_to_pid_to_cond_to_matrices

def get_template_to_pid_to_cond_to_matrix_with_json(light_id, in_json_dir = "/scratch/gpfs/rk1593/clustering_output/"):
    template_to_pid_to_cond_to_matrices = {}
    with open(in_json_dir + "searchlights_matrices/" + light_id + ".json") as json_file:
        template_to_pid_to_cond_to_matrices = json.load(json_file)
    return template_to_pid_to_cond_to_matrices


def get_template_to_pid_to_cond_to_matrix_new(tar_file_dir, light_id):
    os.chdir(tar_file_dir)
    template_to_pid_to_cond_to_matrices = {}
    tar_file_name = light_id + "_new.tar.gz"
    tar = tarfile.open(tar_file_name)
    files_this_light = [x.name for x in tar.getmembers()]
    num_files_this_light = len(files_this_light)
    if num_files_this_light != 480:
        print("Error: searchlight " + light_id + " has " + str(num_files_this_light) + " files." )
        return
    for file_name in files_this_light:
        splitted_file_name = file_name.split("_")
        # get the participant id
        pid = splitted_file_name[1]
        cond = splitted_file_name[3] + "-" + splitted_file_name[4]
        template_id = file_name[-5]
        in_text = tar.extractfile(file_name).read()
        csv_file = io.StringIO(in_text.decode('ascii'))
        random_replacer_for_nothing = str(3e200) 
        csv_lines = [[y.replace(" ", "") for y in x] for x in csv.reader(csv_file)]
        csv_lines = [[y if y != "" else random_replacer_for_nothing for y in x] for x in csv_lines]
        new_arr = np.array(csv_lines).astype("float")
        new_arr[abs(new_arr) > 1] = np.nan
        mean_list = np.nanmean(new_arr, axis = 0).tolist()
        if np.count_nonzero(np.isnan(mean_list)) != 0:
            print("Error: mean list has np.nan")
            print("mean_list: ", mean_list)
            # if isnan has all Falses and so everything is 0
            return 
        if template_id not in template_to_pid_to_cond_to_matrices:
            template_to_pid_to_cond_to_matrices[template_id] = {}
        if pid not in template_to_pid_to_cond_to_matrices[template_id]:
            template_to_pid_to_cond_to_matrices[template_id][pid] = {}
        template_to_pid_to_cond_to_matrices[template_id][pid][cond] = new_arr
    return template_to_pid_to_cond_to_matrices

def get_template_to_pid_to_cond_to_lists(light_id, tar_file_dir, in_json_dir):
    with open(in_json_dir + "searchlights_lists/" + light_id + ".json") as json_file:
        template_to_pid_to_cond_to_lists = json.load(json_file)

    if len(template_to_pid_to_cond_to_lists) != 0:
        return template_to_pid_to_cond_to_lists
    else:
        print("missing ", light_id)
        template_to_pid_to_cond_to_lists = {}
        os.chdir(tar_file_dir)
        tar_file_name = light_id + "_new.tar.gz"
        tar = tarfile.open(tar_file_name)
        files_this_light = [x.name for x in tar.getmembers()]
        template2_count = 0
        template3_count = 0
        template4_count = 0
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
            in_text = tar.extractfile(file_name).read()
            csv_file = io.StringIO(in_text.decode('ascii'))
            random_replacer_for_nothing = str(3e200) # this is set to be something super big (greater than 1), so that 
            csv_lines = [[y.replace(" ", "") for y in x] for x in csv.reader(csv_file)]
            csv_lines = [[y if y != "" else random_replacer_for_nothing for y in x] for x in csv_lines]
            try:
                new_arr = np.array(csv_lines).astype("float")
                new_arr[abs(new_arr) > 1] = np.nan
            except ValueError:
                pdb.set_trace()       
            mean_list = np.nanmean(new_arr, axis = 0).tolist()
            if template_id not in template_to_pid_to_cond_to_lists:
                template_to_pid_to_cond_to_lists[template_id] = {}
            if pid not in template_to_pid_to_cond_to_lists[template_id]:
                template_to_pid_to_cond_to_lists[template_id][pid] = {}
            template_to_pid_to_cond_to_lists[template_id][pid][cond] = mean_list
    return template_to_pid_to_cond_to_lists

def interpret(cond, template_id, event_match_id):
    # if we didn't get one of the easy mappings
    if cond == 'sameEv-sameSchema':
        return 'correct-path'
    elif cond == 'otherEv-sameSchema':
        return 'other-path-from-same-schema'

    # then do the conditionals
    # if event2 is used as template, then:
    # - during event2 TRs the same-event/other-schema condition is the visually-matched-path and the other-event/other-schema condition is the unrelated-path.
    # - during event3 TRs the other-event/other-schema condition is the visually-matched-path and the same-event/other-schema condition is the unrelated-path.
    # - during event4 TRs the same-event/other-schema condition is the visually-matched-path and the other-event/other-schema condition is the unrelated-path.
    if template_id == 2:
        if event_match_id == 2:
            if cond == 'sameEv-otherSchema':
                return 'visually-matched-path'
            elif cond == 'otherEv-otherSchema':
                return 'unrelated-path'
        elif event_match_id == 3:
            if cond == 'sameEv-otherSchema':
                return 'unrelated-path'
            elif cond == 'otherEv-otherSchema':
                return 'visually-matched-path'
        elif event_match_id == 4:
            if cond == 'sameEv-otherSchema':
                return 'visually-matched-path'
            elif cond == 'otherEv-otherSchema':
                return 'unrelated-path'
        else:
            print("Error: event_match_id invalid", event_match_id)
    # if event3 is used as template, then:
    # - during event2 TRs the other-event/other-schema condition is the visually-matched-path and the same-event/other-schema condition is the unrelated-path.
    # - during event3 TRs the same-event/other-schema condition is the visually-matched-path and the other-event/other-schema condition is the unrelated-path.
    # - during event4 TRs the other-event/other-schema condition is the visually-matched-path and the same-event/other-schema condition is the unrelated-path.
    elif template_id == 3:
        if event_match_id == 2:
            if cond == 'sameEv-otherSchema':
                return 'unrelated-path'
            elif cond == 'otherEv-otherSchema':
                return 'visually-matched-path'
        elif event_match_id == 3:
            if cond == 'sameEv-otherSchema':
                return 'visually-matched-path'
            elif cond == 'otherEv-otherSchema':
                return 'unrelated-path'
        elif event_match_id == 4:
            if cond == 'sameEv-otherSchema':
                return 'unrelated-path'
            elif cond == 'otherEv-otherSchema':
                return 'visually-matched-path'
        else:
            print("Error: event_match_id invalid", event_match_id)

    # if event4 is used as template, then:
    # - during event2 TRs the same-event/other-schema condition is the visually-matched-path and the other-event/other-schema condition is the unrelated-path.
    # - during event3 TRs the other-event/other-schema condition is the visually-matched-path and the same-event/other-schema condition is the unrelated-path.
    # - during event4 TRs the same-event/other-schema condition is the visually-matched-path and the other-event/other-schema condition is the unrelated-path.
    elif template_id == 4:
        if event_match_id == 2:
            if cond == 'sameEv-otherSchema':
                return 'visually-matched-path'
            elif cond == 'otherEv-otherSchema':
                return 'unrelated-path'
        elif event_match_id == 3:
            if cond == 'sameEv-otherSchema':
                return 'unrelated-path'
            elif cond == 'otherEv-otherSchema':
                return 'visually-matched-path'
        elif event_match_id == 4:
            if cond == 'sameEv-otherSchema':
                return 'visually-matched-path'
            elif cond == 'otherEv-otherSchema':
                return 'unrelated-path'
        else:
            print("Error: event_match_id invalid", event_match_id)
    else:
        print(type(template_id))
        print(event_match_id)
        print(cond)
        print("Error: template_id invalid", template_id)


def create_raw_df_clusters_fingerprint_to_plot(cluster_to_template_to_cond_to_mean_list):
    cluster_id_list = []
    condition_list = []
    tr_list = []
    corr_list = []
    matching_id_list = []
    template_id_list = []
    std_list = []
    n_list = []
    CI_lb_list = []
    CI_ub_list = []
    matching_id_labels_each_tr = []
    for i in range(0,17):
        matching_id_labels_each_tr.append(0)
    for i in range(17,23):
        matching_id_labels_each_tr.append(1)
    for i in range(23,34):
        matching_id_labels_each_tr.append(2)
    for i in range(34,50):
        matching_id_labels_each_tr.append(3)
    for i in range(50,66):
        matching_id_labels_each_tr.append(4)
    for i in range(66,74):
        matching_id_labels_each_tr.append(5)

    for cluster_id in cluster_to_template_to_cond_to_mean_list:
        for template_id in cluster_to_template_to_cond_to_mean_list[cluster_id]:
            for cond in cluster_to_template_to_cond_to_mean_list[cluster_id][template_id]:
                tr_num = 0
                current_match_id = matching_id_labels_each_tr[0]
                for index,value in enumerate(cluster_to_template_to_cond_to_mean_list[cluster_id][template_id][cond]["mean_list"]):
                    match_id = matching_id_labels_each_tr[index]
                    if current_match_id != match_id:
                        current_match_id = match_id
                        tr_num = 0
                    matching_id_list.append(match_id)
                    tr_list.append(tr_num)
                    tr_num += 1
                    corr_list.append(value)
                    CI_lb, CI_ub = cluster_to_template_to_cond_to_mean_list[cluster_id][template_id][cond]["lb_ub_list"][index]
                    CI_lb_list.append(CI_lb)
                    CI_ub_list.append(CI_ub)
                    std = cluster_to_template_to_cond_to_mean_list[cluster_id][template_id][cond]["std_list"][index]
                    std_list.append(std)
                    n_list.append(cluster_to_template_to_cond_to_mean_list[cluster_id][template_id][cond]["n"])
                    template_id_list.append(template_id)
                    cluster_id_list.append(cluster_id)      
                    new_cond = cond #interpret(cond, template_id, match_id)
                    condition_list.append(new_cond)


# The same-event/same-schema condition is always the correct-path condition, and the other-event/same-schema condition is always the other-path-from-same-schema condition.

    output_dict = {"tr": tr_list,
                   "path_id": condition_list,
                   "event_template_id": template_id_list,
                   "cluster_id": cluster_id_list,
                   "corr":corr_list,
                   "std": std_list,
                   "n":n_list, # we should n be the same for every cluster, check that!
                   "lower_bound":CI_lb_list,
                   "upper_bound":CI_ub_list,
                   "event_matching_id":matching_id_list }
    return pd.DataFrame(output_dict)

def get_b(n, s, x):
    lb = x - (s / math.sqrt(n))
    ub = x + (s / math.sqrt(n))
    return (lb, ub)

def process_cluster(return_dict, cluster_label, df, cluster_id_to_distilled_lights, condition_types = 
                            ['sameEv-sameSchema', 'sameEv-otherSchema', 
                            'otherEv-sameSchema', 'otherEv-otherSchema']):
    temp_dict = {}
    if cluster_id_to_distilled_lights == None:
        light_list = df[df["cluster_assignment"] == cluster_label]["searchlight"].tolist()
    else:
        light_list = cluster_id_to_distilled_lights[cluster_label]
    # first get the N searchlight fingerprint plots for each of the 40 subject fingerprint plots
    pid_to_template_to_cond_to_matrices_per_light = {}
    unique_pids_list = []
    for light_index,light_id in enumerate(light_list):
        if light_index % 1000 == 0:
            print("cluster_label: ",cluster_label ,", light_index: ", light_index)
        template_to_pid_to_cond_to_matrix = get_template_to_pid_to_cond_to_matrix(light_id)
        for template_id in [2,3,4]:
            for pid in template_to_pid_to_cond_to_matrix[template_id]:
                if pid not in unique_pids_list:
                    unique_pids_list.append(pid)
                if pid not in pid_to_template_to_cond_to_matrices_per_light:
                    pid_to_template_to_cond_to_matrices_per_light[pid] = {}
                if template_id not in pid_to_template_to_cond_to_matrices_per_light[pid]:
                    pid_to_template_to_cond_to_matrices_per_light[pid][template_id] = {}
                for cond in template_to_pid_to_cond_to_matrix[template_id][pid]:
                    if cond not in pid_to_template_to_cond_to_matrices_per_light[pid][template_id]:
                        pid_to_template_to_cond_to_matrices_per_light[pid][template_id][cond] = []
                    new_arr = np.array(template_to_pid_to_cond_to_matrix[template_id][pid][cond])
                    nrow,ncol = np.shape(new_arr)
                    if light_id  == "48_60_70" and pid == "sub-102" and template_id == 2:
                        print("new_arr: ", new_arr)
                    if nrow != 12 or ncol != 74:
                        print("Error: nrow != 12 or ncol != 74")
                        return
                    pid_to_template_to_cond_to_matrices_per_light[pid][template_id][cond].append(new_arr)
    # check that have 40 participants
    if len(unique_pids_list) != 40:
        print("Error: unique_pids_list != 40")
        print(unique_pids_list)
        return

    for template_id in [2,3,4]:
        if template_id not in temp_dict:
            temp_dict[template_id] = {}
        for cond in condition_types:
            this_cluster_template_condition_list_of_lists = []
            # stack up all 40 mean lists, one from each subject
            for pid in unique_pids_list:
                if len(pid_to_template_to_cond_to_matrices_per_light[pid][template_id][cond]) != len(light_list):        
                    print("Error: the number of matrices to average over is not equal to number of searchlights.")
                    print("len(pid_to_template_to_cond_to_matrices_per_light[pid][template_id][cond]): ", len(pid_to_template_to_cond_to_matrices_per_light[pid][template_id][cond]))
                    print("len(light_list): ", len(light_list))
                    return 
                # here is where we get the subjects' fingerprint plot across all searchlights and weddings
                big_matrix = np.vstack(pid_to_template_to_cond_to_matrices_per_light[pid][template_id][cond])
                # no longer needed so remove from memory
                pid_to_template_to_cond_to_matrices_per_light[pid][template_id][cond] = 0
                subject_mean_list  = np.nanmean(big_matrix, axis = 0)
                if np.count_nonzero(np.isnan(subject_mean_list)) != 0 or len(subject_mean_list) != 74:
                    print("Error: np.count_nonzero(np.isnan(mean_list)) != 0 or len(mean_list) != 74")
                    print(subject_mean_list)
                    return
                this_cluster_template_condition_list_of_lists.append(subject_mean_list)
            # at this point we have the information from the 40 subject fingerprint plots
            cluster_template_condition_matrix = np.vstack(this_cluster_template_condition_list_of_lists)
            # get the cluster average mean list across the 40 participants
            cluster_mean_list = np.nanmean(cluster_template_condition_matrix, axis = 0)
            std_list = np.nanstd(cluster_template_condition_matrix, axis = 0)
            n = len(cluster_template_condition_matrix) # this should be 40 always
            if n != 40:
                print("Error: n != 40")
                return
            lb_ub_list = [get_b(n,std,mean) for mean,std in zip(cluster_mean_list, std_list)]
            temp_dict[template_id][cond] = {"mean_list":cluster_mean_list,
                                            "std_list": std_list,
                                            "n":n,
                                            "lb_ub_list": lb_ub_list}

    return_dict[cluster_label] = temp_dict

# requires a df with columns searchlight, features and cluster assignment
def from_cluster_labels_to_fingerprint_avg_df(df, cluster_id_to_distilled_lights):
    manager = Manager()
    return_dict = manager.dict()
    # for each cluster label
    processes_list = []
    for cluster_label in set(df["cluster_assignment"].tolist()):
        print("cluster_label: ", cluster_label)
        new_process = Process(target= process_cluster, args = (return_dict, cluster_label,df, cluster_id_to_distilled_lights))
        new_process.start()
        processes_list.append(new_process)
    for p in processes_list:
        p.join()
    cluster_to_template_to_cond_to_mean_list = {}
    for key in return_dict:
        cluster_to_template_to_cond_to_mean_list[key] = return_dict[key]
    return create_raw_df_clusters_fingerprint_to_plot(cluster_to_template_to_cond_to_mean_list)


def get_cluster_id_to_distilled_searchlights(K, distill_numLights,
                      exaggerated_0, weight_it, cap_zero, fraction):
    cluster_id_to_distilled_lights = {}
    for cluster_id in range(1,K+1):
        ranks_dir = "/scratch/gpfs/rk1593/clustering_output/kmeans_searchlight_ranks/K" + str(K) + "/kmeans" + str(K) + "cluster" + str(cluster_id) + "_ranks"
        if exaggerated_0:
            ranks_dir += "_Consider0"
        else:
            ranks_dir += "_NoConsider0"
        if weight_it:
            ranks_dir += "_WeightByMean"
        else:
            ranks_dir += "_NoWeightByMean"
        if cap_zero:
            ranks_dir += "_CapZero"
        else:
            ranks_dir += "_NoCapZero"
        ranks_dir += ".csv"
        ranks_df = pd.read_csv(ranks_dir)
        ranked_searchlights_in_cluster = ranks_df["searchlights_in_cluster"].tolist()
        if fraction:
            cluster_id_to_distilled_lights[cluster_id] = ranked_searchlights_in_cluster[0:int(distill_numLights*len(ranked_searchlights_in_cluster))]
        else: 
            cluster_id_to_distilled_lights[cluster_id] = ranked_searchlights_in_cluster[0:distill_numLights]

    return cluster_id_to_distilled_lights


job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
K = job_id_in
cluster_assignment_csv_path = "/scratch/gpfs/rk1593/clustering_output/kmeans_assignments_tval/kmeans_" + str(K) + "clusters_tval.csv"
exaggerated_0 = False
weight_it = True
cap_zero = False
distill_it = False
distill_numLights = 0.5
fraction = True
optional_focus = None
if distill_it:  
    cluster_id_to_distilled_lights = get_cluster_id_to_distilled_searchlights(K, distill_numLights, exaggerated_0, weight_it, cap_zero, fraction)
else:
    cluster_id_to_distilled_lights = None
assignment_df = pd.read_csv(cluster_assignment_csv_path)
in_json_dir = "/scratch/gpfs/rk1593/clustering_output/"
raw_df = from_cluster_labels_to_fingerprint_avg_df(assignment_df, cluster_id_to_distilled_lights)
out_path = "/scratch/gpfs/rk1593/clustering_output/kmeans_fingerprints_tval/kmeans_" + str(K) + "clusters_tval" 
if distill_it:
    if exaggerated_0:
        out_path += "_Consider0"
    else:
        out_path += "_NoConsider0"
    if weight_it:
        out_path += "_WeightByMean"
    else:
        out_path += "_NoWeightByMean"
    if cap_zero:
        out_path += "_CapZero"
    else:
        out_path += "_NoCapZero"
    out_path += ("_" + str(distill_numLights) + "distill_numLights")
out_path += ".csv"
raw_df.to_csv(out_path)

