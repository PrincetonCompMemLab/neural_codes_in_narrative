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
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples
from joblib import dump, load
from sklearn.cluster import AgglomerativeClustering

# dump(clf, 'filename.joblib') 
# clf = load('filename.joblib') 
def agg_clustering(dir, K, n_jobs):
    X = np.load(file = dir + "kmodes_assignments/master_X_list.npy")
    searchlight_list = np.load(file = dir + "kmodes_assignments/master_searchlight_list.npy").tolist()
    X = X.astype("float32") # for memory limits
    print("got X")
    D_matrix = pairwise_distances(X, metric='hamming', n_jobs= n_jobs)
    X = 0 # for memory
    model = AgglomerativeClustering(distance_threshold=0, n_clusters = K, linkage = "average", affinity = "precomputed")
    model.fit(D_matrix)
    cluster_labels = model.labels_
    # add 1 so that we have no label with zero
    cluster_labels += 1
    output_dict = {"cluster_assignment": cluster_labels.tolist(),
                "searchlight": searchlight_list
                }
    df = pd.DataFrame(output_dict)
    df.to_csv(dir + "kmodes_assignments/agglomerative_" + str(K) + "clusters.csv")
    sample_silhouette_values = silhouette_samples(X = D_matrix, labels = cluster_labels, metric = 'precomputed')
    print("got s_coeffs")
    np.savetxt(dir + "/kmodes_silhouette/agglomerative" + str(K) + "_silhouttes.csv", sample_silhouette_values, delimiter=",")


job_id_in = int(os.environ["SLURM_ARRAY_TASK_ID"])
dir = "/scratch/gpfs/rk1593/clustering_output/" 
K = job_id_in
n_jobs = 32
agg_clustering(dir = dir, K = K, n_jobs = n_jobs)