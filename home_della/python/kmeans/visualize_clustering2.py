import enum
from itertools import count
from logging import raiseExceptions
import os
import tarfile
import pdb
import csv
import io
from scipy.stats.morestats import Std_dev
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from statistics import stdev
import json
import time
import math
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.spatial import distance
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import wasserstein_distance, pearsonr, spearmanr
from plotnine import ggplot, ggtitle, labs, geom_line,geom_label, geom_point, aes, geom_text, stat_smooth, facet_wrap, geom_boxplot, geom_histogram, xlim, ylim, facet_grid
from kmodes.kmodes import KModes
import nibabel as nib
from multiprocessing import Process
from multiprocessing import Pool 
from multiprocessing import Manager
from nilearn import plotting
from nilearn import image
import nibabel as nib
import numpy as np
import os

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
    ['sameEv-sameSchema', 'sameEv-otherSchema', 
                            'otherEv-sameSchema', 'otherEv-otherSchema']
    for cluster_id in cluster_to_template_to_cond_to_mean_list:
        for template_id in cluster_to_template_to_cond_to_mean_list[cluster_id]:
            for cond in cluster_to_template_to_cond_to_mean_list[cluster_id][template_id]:
                tr_num = 0
                current_match_id = matching_id_labels_each_tr[0]
                for index,value in enumerate(cluster_to_template_to_cond_to_mean_list[cluster_id][template_id][cond]):
                    match_id = matching_id_labels_each_tr[index]
                    if current_match_id != match_id:
                        current_match_id = match_id
                        tr_num = 0
                    matching_id_list.append(match_id)
                    tr_list.append(tr_num)
                    tr_num += 1
                    corr_list.append(value)
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
                   "event_matching_id":matching_id_list }
    return pd.DataFrame(output_dict)

def process_cluster(return_dict, cluster_label, df,tar_file_dir, in_json_dir,  condition_types = 
                            ['sameEv-sameSchema', 'sameEv-otherSchema', 
                            'otherEv-sameSchema', 'otherEv-otherSchema']):
    temp_dict = {}
    light_list = df[df["cluster_assignment"] == cluster_label]["searchlight"].tolist()
    # now we create list of lists where the list of lists contains lists in different searchlights
    # but with the sample cluster, template, pid (participant id), and condition
    for template_id in ['2','3','4']:
        if template_id not in temp_dict:
            temp_dict[template_id] = {}
        for cond in condition_types:
            this_cluster_template_condition_list_of_lists = []
            for index,light_id in enumerate(light_list):
                if index % 10000 == 0:
                    print("template_id: ", str(template_id), ", cond: ", cond, ", cluster_label: ", cluster_label, ", light_id index: ", index)
                template_to_pid_to_cond_to_lists = get_template_to_pid_to_cond_to_lists(light_id, tar_file_dir, in_json_dir)
                for pid in template_to_pid_to_cond_to_lists[template_id]:
                    this_cluster_template_condition_list_of_lists.append(template_to_pid_to_cond_to_lists[template_id][pid][cond])
            cluster_template_condition_matrix = np.vstack(this_cluster_template_condition_list_of_lists)
            mean_list = np.nanmean(cluster_template_condition_matrix, axis = 0)
            temp_dict[template_id][cond] = mean_list
            print("np.shape(mean_list): ",np.shape(mean_list))
    return_dict[cluster_label] = temp_dict

# requires a df with columns searchlight, features and cluster assignment
def from_cluster_labels_to_fingerprint_avg_df(df, tar_file_dir, in_json_dir,  condition_types = 
                            ['sameEv-sameSchema', 'sameEv-otherSchema', 
                            'otherEv-sameSchema', 'otherEv-otherSchema']):
    manager = Manager()
    return_dict = manager.dict()
    # for each cluster label
    processes_list = []
    for cluster_label in set(df["cluster_assignment"].tolist()):
        new_process = Process(target= process_cluster, args = (return_dict, cluster_label,df,
                                                            tar_file_dir, in_json_dir, condition_types))
        new_process.start()
        processes_list.append(new_process)
    for p in processes_list:
        p.join()
    cluster_to_template_to_cond_to_mean_list = {}
    for key in return_dict:
        cluster_to_template_to_cond_to_mean_list[key] = return_dict[key]
    return create_raw_df_clusters_fingerprint_to_plot(cluster_to_template_to_cond_to_mean_list)

def plot_raw_fingerprint_cluster(raw_df, cluster_id, event_template_id):
    filtered_df = raw_df[(raw_df["cluster_id"] == cluster_id) & (raw_df["event_template_id"] == event_template_id)]
    (ggplot(filtered_df) 
         + geom_line(aes(x="tr", y="corr", color = "factor(path_id)"))
             + facet_grid('.~ event_matching_id')
            + labs(x='tr', y='R')
             + ggtitle("Cluster " + str(cluster_id) +  "& Event Template " + str(event_template_id) )
    ).draw()


job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
n_clusters = job_id_in
cluster_assignment_csv_path = "/scratch/gpfs/rk1593/clustering_output/kmodes_assignments/kmodes_" + str(n_clusters) + "clusters.csv"
assignment_df = pd.read_csv(cluster_assignment_csv_path)
tar_file_dir = "/scratch/gpfs/rk1593/tar_by_searchlight/tar_by_searchlight/"
in_json_dir = "/scratch/gpfs/rk1593/clustering_output/"
raw_df = from_cluster_labels_to_fingerprint_avg_df(assignment_df, tar_file_dir, in_json_dir)
raw_df.to_csv("/scratch/gpfs/rk1593/clustering_output/kmodes_fingerprints/kmodes_" + str(n_clusters) + "clusters.csv")