import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples
import os
import pandas as pd

num_jobs = 32
dir = "/scratch/gpfs/rk1593/clustering_output/" 
job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
K = job_id_in
labels_dir = "/scratch/gpfs/rk1593/clustering_output/kmeans_assignments_tval/kmeans_" + str(K) + "clusters_tval.csv"
df = pd.read_csv(labels_dir)
labels = df["cluster_assignment"].tolist()
df = 0 # for memory
output_dir = "/scratch/gpfs/rk1593/clustering_output/kmeans_silhouette_tval/kmeans" + str(K) + "_silhouttes_tval.csv"
X = np.load("/scratch/gpfs/rk1593/clustering_output/kmeans_assignments_tval/master_X_list_tval.npy")
X = X.astype("float32") # for memory limits
print("got X")
D_matrix = pairwise_distances(X, n_jobs= num_jobs)
print("got D_matrix")
X = 0
sample_silhouette_values = silhouette_samples(X = D_matrix, labels = labels, metric = 'precomputed')
print("got s_coeffs")
np.savetxt(output_dir, sample_silhouette_values, delimiter=",")
