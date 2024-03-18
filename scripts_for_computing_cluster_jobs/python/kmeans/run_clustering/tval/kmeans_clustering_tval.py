
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans


def kmeans_clustering(dir, K):
    # read in the master_X_list and master_searchlight so that we have the cluster assignments
    # for the searchlights without having to output so much data each time
    # by outputting the master_X_list to the output csv multiple times
    # which would be a waste of della's memory
    X = np.load(file = dir + "master_X_list_tval.npy")
    searchlight_list = np.load(file = dir + "master_searchlight_tval.npy").tolist()
    print("K_clusters: ", K)
    kmeans = KMeans(n_clusters=K)
    cluster_labels = kmeans.fit_predict(X)
    # add 1 so that we have no label with zero
    cluster_labels += 1
    # create csv with the cluster assignment for this number of clusters
    # create the output df by giving cluster labels
    this_cost = kmeans.inertia_
    cost_list = [this_cost for i in range(len(searchlight_list))]
    output_dict = {"cluster_assignment": cluster_labels.tolist(),
                "searchlight": searchlight_list,
                "cost":cost_list
                }
    df = pd.DataFrame(output_dict)
    df.to_csv(dir + "kmeans_" + str(K) + "clusters_tval.csv")
    # It is important to use binary access
    with open(dir + "kmeans_" + str(K) + "clusters_tval.pickle", 'wb') as f:
        pickle.dump(kmeans, f)
  

job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
dir = "/scratch/gpfs/rk1593/clustering_output/kmeans_assignments_tval/"
K = job_id_in
kmeans_clustering(dir = dir, K = K)