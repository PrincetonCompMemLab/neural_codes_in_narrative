import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples
import os
import pandas as pd

num_jobs = 32
dir = "/scratch/gpfs/KNORMAN/rkempner/clustering_output/" 
job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
K = job_id_in
labels_dir = f"{dir}/kmeans_assignments_just_means/kmeans_" + str(K) + "clusters_just_means.csv"
df = pd.read_csv(labels_dir)
labels = df["cluster_assignment"].tolist()
df = 0 # for memory
output_dir = f"{dir}/kmeans_silhouette_just_means/kmeans" + str(K) + "_silhouttes_just_means.csv"
X = np.load(f"{dir}/kmeans_assignments_just_means/master_X_list_just_means.npy")
X = X.astype("float32") # for memory limits
print("got X")
D_matrix = pairwise_distances(X, n_jobs= num_jobs)
print("got D_matrix")
X = 0
sample_silhouette_values = silhouette_samples(X = D_matrix, labels = labels, metric = 'precomputed')
print("got s_coeffs")
np.savetxt(output_dir, sample_silhouette_values, delimiter=",")
