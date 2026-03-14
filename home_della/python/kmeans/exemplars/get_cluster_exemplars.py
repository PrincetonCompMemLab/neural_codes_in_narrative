import numpy as np
import os
import pandas as pd
import pdb 
from multiprocessing import Process

def get_rank(mean_vec, test_vec, weight_it = True, cap_zero = False):
    rank_vec = []
    for index, mean_val in enumerate(mean_vec):
        test_val = test_vec[index]
        if np.sign(mean_val) == np.sign(test_val):
            rank_val = abs(test_val) - abs(mean_val)
        # different signs
        else:
            if cap_zero:
                rank_val = -1 * (abs(mean_val))
            else:
                rank_val = -1 * (abs(test_val) + abs(mean_val))
        rank_vec.append(rank_val)
    # now weight the rank vector by abs(mean_vec)
    if weight_it:
        return np.dot(rank_vec, np.abs(mean_vec))
    # otherwise just sum it
    return sum(rank_vec)

# this assumes that when the mean is close enough to to zero
# then values close to zero should be rewarded to get our exemplars
# basically it identifies each data point as separate or together in the mean
# and then either wants exaggerated separateness or exaggerated closeness 
def get_rank_exaggerated_zero(mean_vec, test_vec, threshold, cap_zero = False):
    rank_vec = []
    for index, mean_val in enumerate(mean_vec):
        test_val = test_vec[index]
        # when greater than the theshold then we 
        # want to try to award tvalues which are more exaggeratted in the 
        # direction of separation
        if abs(mean_val) > threshold:
            if np.sign(mean_val) == np.sign(test_val):
                rank_val = abs(test_val) - abs(mean_val)
            # different signs
            else:
                if cap_zero:
                    rank_val = -1 * (abs(mean_val))
                else:
                    rank_val = -1 * (abs(test_val) + abs(mean_val))
            rank_vec.append(rank_val)
        # then we want exaggerated closeness to zero, so penalize away from zero
        else:
            rank_val = (-1 * abs(test_val))
   
    return sum(rank_vec)


def sort_list(ranks, lights):
    zipped_pairs = zip(ranks, lights)
    new_ranks = []
    new_lights = []
    for r,l in sorted(zipped_pairs, reverse= True):
        new_ranks.append(r)
        new_lights.append(l)
    return new_ranks, new_lights

def process_cluster(cluster_id, X, cluster_labels, searchlight_list, 
                exaggerated_closeness, weight_it, cap_zero, threshold):
    print("cluster_id: ", cluster_id)
    vectors_in_cluster = X[cluster_labels == cluster_id,:]
    searchlights_in_cluster = searchlight_list[cluster_labels == cluster_id]
    tvalue_vectors_centroid = np.mean(vectors_in_cluster, axis = 0)
    # now go through each searchlight and get its rank
    searchlights_rank_list = []
    for index,test_vec in enumerate(vectors_in_cluster):
        if index % 10000 == 0:
            print(index)
        if exaggerated_closeness:
            new_rank = get_rank_exaggerated_zero(tvalue_vectors_centroid, test_vec, threshold, cap_zero)
            searchlights_rank_list.append(new_rank)
        else:
            new_rank = get_rank(tvalue_vectors_centroid, test_vec, weight_it, cap_zero)
            searchlights_rank_list.append(new_rank)
    # do some sorting by rank including both lists and then output df
    new_ranks, new_lights = sort_list(ranks = searchlights_rank_list, lights = searchlights_in_cluster)
    output_dir = "/scratch/gpfs/rk1593/clustering_output/kmeans_searchlight_ranks/K" + str(K) + "/kmeans" + str(K) + "cluster" + str(cluster_id) + "_ranks"
    if exaggerated_closeness:
        output_dir += "_Consider0"
    else:
        output_dir += "_NoConsider0"
    if weight_it:
        output_dir += "_WeightByMean"
    else:
        output_dir += "_NoWeightByMean"
    if cap_zero:
        output_dir += "_CapZero"
    else:
        output_dir += "_NoCapZero"
    output_dir += ".csv"
    df = pd.DataFrame({"searchlights_rank": new_ranks,
                       "searchlights_in_cluster":new_lights})
    df.to_csv(output_dir)

# for each cluster, get the mean of that cluster
job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
K = job_id_in
labels_dir = "/scratch/gpfs/rk1593/clustering_output/kmeans_assignments_tval/kmeans_" + str(K) + "clusters_tval.csv"
df = pd.read_csv(labels_dir)
cluster_labels = df["cluster_assignment"]
searchlight_list = np.load("/scratch/gpfs/rk1593/clustering_output/kmeans_assignments_tval/master_searchlight_tval.npy")
X = np.load("/scratch/gpfs/rk1593/clustering_output/kmeans_assignments_tval/master_X_list_tval.npy")
exaggerated_closeness = False
weight_it = True
cap_zero = False
threshold = 0.1

processes_list = []
for cluster_id in set(cluster_labels):
    new_p = Process(target = process_cluster, args= (cluster_id, X, cluster_labels, searchlight_list, 
                exaggerated_closeness, weight_it, cap_zero, threshold) )
    new_p.start()
    processes_list.append(new_p)

for p in processes_list:
    p.join()