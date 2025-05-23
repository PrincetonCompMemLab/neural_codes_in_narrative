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


def create_raw_df_clusters_fingerprint_to_plot(cluster_to_template_to_cond_to_event_to_violin):
    cluster_id_list = []
    condition_list = []
    mean_each_pid_list = []
    matching_id_list = []
    template_id_list = []
    pid_list = []

    for cluster_id in cluster_to_template_to_cond_to_event_to_violin:
        for template_id in cluster_to_template_to_cond_to_event_to_violin[cluster_id]:
            for cond in cluster_to_template_to_cond_to_event_to_violin[cluster_id][template_id]:
                    for event in cluster_to_template_to_cond_to_event_to_violin[cluster_id][template_id][cond]:
                        info_dict = cluster_to_template_to_cond_to_event_to_violin[cluster_id][template_id][cond][event]
                        for this_event_mean_this_participant, pid in zip (info_dict["collapseTR_mean_each_pid"], info_dict["pids_list"]):
                            mean_each_pid_list.append(this_event_mean_this_participant)
                            pid_list.append(pid)
                            cluster_id_list.append(cluster_id)
                            template_id_list.append(template_id)
                            condition_list.append(cond)
                            matching_id_list.append(event)
    # The same-event/same-schema condition is always the correct-path condition, and the other-event/same-schema condition is always the other-path-from-same-schema condition.
    output_dict = {
                   "cluster_id": cluster_id_list,
                   "event_template_id": template_id_list,
                   "event_matching_id":matching_id_list,
                   "path_id": condition_list,
                   "pid": pid_list,
                  "mean_each_pid": mean_each_pid_list}
    return pd.DataFrame(output_dict)

def get_b(n, s, x):
    lb = x - (s / math.sqrt(n))
    ub = x + (s / math.sqrt(n))
    return (lb, ub)

def process_cluster(return_dict, cluster_label, df, cluster_id_to_distilled_lights, condition_types = 
                            ['sameEv-sameSchema', 'sameEv-otherSchema', 
                            'otherEv-sameSchema', 'otherEv-otherSchema']):
    temp_dict = {}
    event_1_start = 17
    event_2_start = 23
    event_3_start = 34
    event_4_start = 50
    event_5_start = 66
    if cluster_id_to_distilled_lights == None:
        light_list = df[df["cluster_assignment"] == cluster_label]["searchlight"].tolist()
    else:
        light_list = cluster_id_to_distilled_lights[cluster_label]
    # first get the N searchlight fingerprint plots for each of the 40 subject fingerprint plots
    pid_to_template_to_cond_to_event_to_mean_per_light = {}
    unique_pids_list = []
    print(f"cluster label {cluster_label}, len(light_list): {len(light_list)}")
    for light_index,light_id in enumerate(light_list):
        if light_index % 1000 == 0:
            print("cluster_label: ",cluster_label ,", light_index: ", light_index)
        template_to_pid_to_cond_to_matrix = get_template_to_pid_to_cond_to_matrix_with_json(light_id)
        for template_id in [2,3,4]:
            for pid in template_to_pid_to_cond_to_matrix[str(template_id)]:
                if pid not in unique_pids_list:
                    unique_pids_list.append(pid)
                if pid not in pid_to_template_to_cond_to_event_to_mean_per_light:
                    pid_to_template_to_cond_to_event_to_mean_per_light[pid] = {}
                if template_id not in pid_to_template_to_cond_to_event_to_mean_per_light[pid]:
                    pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id] = {}
                for cond in template_to_pid_to_cond_to_matrix[str(template_id)][pid]:
                    if cond not in pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id]:
                        pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id][cond] = {}
                    new_arr = np.array(template_to_pid_to_cond_to_matrix[str(template_id)][pid][cond])
                    mean_all_events = np.nanmean(new_arr, axis = 0).tolist()
                    mean_event2 = np.nanmean(mean_all_events[event_2_start:event_3_start])
                    mean_event3 = np.nanmean(mean_all_events[event_3_start:event_4_start])
                    mean_event4 = np.nanmean(mean_all_events[event_4_start:event_5_start])
                    mean_events_list = [mean_event2, mean_event3, mean_event4]
                    nrow,ncol = np.shape(new_arr)
                    if light_id  == "48_60_70" and pid == "sub-102" and template_id == 2:
                        print("new_arr: ", new_arr)
                    if nrow != 12 or ncol != 74:
                        print("Error: nrow != 12 or ncol != 74")
                        return
                    for event_idx, event in enumerate([2,3,4]):
                        if event not in pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id][cond]:
                            pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id][cond][event] = [] 
                        pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id][cond][event].append(mean_events_list[event_idx])
    # check that have 40 participants
    if len(unique_pids_list) != 40:
        print("Error: unique_pids_list != 40")
        print(unique_pids_list)
        return

    for template_id in [2,3,4]:
        if template_id not in temp_dict:
            temp_dict[template_id] = {}
        for cond in condition_types:
            if cond not in temp_dict[template_id]:
                temp_dict[template_id][cond] = {}
            for event in [2,3,4]:
                this_event_mean_each_pid = []
                # stack up all 40 mean lists, one from each subject
                for pid in unique_pids_list:
                    if len(pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id][cond][event]) != len(light_list):        
                        print("Error: the number of matrices to average over is not equal to number of searchlights.")
                        print("len(pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id][cond][event]): ", len(pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id][cond][event]))
                        print("len(light_list): ", len(light_list))
                        return
                    this_pid_mean = np.nanmean(pid_to_template_to_cond_to_event_to_mean_per_light[pid][template_id][cond][event])
                    this_event_mean_each_pid.append(this_pid_mean)
                # at this point we have the information from the 40 subject fingerprint plots
                # get the cluster average mean list across the 40 participants
                cluster_mean = np.nanmean(this_event_mean_each_pid)
                cluster_std = np.nanstd(this_event_mean_each_pid)
                n = len(this_event_mean_each_pid) # this should be 40 always
                if n != 40:
                    print("Error: n != 40")
                    return
                lb_ub = get_b(n,cluster_std,cluster_mean)
                temp_dict[template_id][cond][event] = {"collapseTR_mean_each_pid":this_event_mean_each_pid, "pids_list": unique_pids_list}

                # temp_dict[template_id][cond][event] = {"collapseTR_mean_each_pid":cluster_mean,
                #                                 "std": cluster_std,
                #                                 "n":n,
                #                                 "lb_ub": lb_ub}

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
    cluster_to_template_to_cond_to_event_to_violin = {}
    for key in return_dict:
        cluster_to_template_to_cond_to_event_to_violin[key] = return_dict[key]
    return create_raw_df_clusters_fingerprint_to_plot(cluster_to_template_to_cond_to_event_to_violin)

def get_cluster_id_to_distilled_searchlights(K, distill_numLights, fraction,by_sil = True):
    cluster_id_to_distilled_lights = {}
    for cluster_id in range(1,K+1):
        ranks_dir = "/scratch/gpfs/rk1593/clustering_output/kmeans_searchlight_ranks_collapseTR/K" + str(K) + "/kmeans" + str(K) + "cluster" + str(cluster_id) + "_ranks_collapseTR"
        ranks_dir += "_by_silhouttes"
        ranks_dir += ".csv"
        print(ranks_dir)
        ranks_df = pd.read_csv(ranks_dir)
        ranked_searchlights_in_cluster = ranks_df["searchlights_in_cluster"].tolist()
        if fraction:
            cluster_id_to_distilled_lights[cluster_id] = ranked_searchlights_in_cluster[0:int(distill_numLights*len(ranked_searchlights_in_cluster))]
        else: 
            cluster_id_to_distilled_lights[cluster_id] = ranked_searchlights_in_cluster[0:distill_numLights]

    return cluster_id_to_distilled_lights


job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
K = job_id_in
cluster_assignment_csv_path = "/scratch/gpfs/rk1593/clustering_output/kmeans_assignments_tval_collapseTR/kmeans_" + str(K) + "clusters_collapseTR.csv"
exaggerated_0 = False
weight_it = True
cap_zero = False
distill_it = False
distill_numLights = 0.05
fraction = True
optional_focus = None
if distill_it:  
    cluster_id_to_distilled_lights = get_cluster_id_to_distilled_searchlights(K, distill_numLights, fraction)
else:
    cluster_id_to_distilled_lights = None
assignment_df = pd.read_csv(cluster_assignment_csv_path)
in_json_dir = "/scratch/gpfs/rk1593/clustering_output/"
raw_df = from_cluster_labels_to_fingerprint_avg_df(assignment_df, cluster_id_to_distilled_lights)
out_path = "/scratch/gpfs/rk1593/clustering_output/kmeans_fingerprints_tval_collapseTR/kmeans_" + str(K) + "clusters_tval_collapseTR" 
if distill_it:
    out_path += "_by_sil"
    out_path += ("_" + str(distill_numLights) + "distill_numLights")
out_path += ".csv"
raw_df.to_csv(out_path)

