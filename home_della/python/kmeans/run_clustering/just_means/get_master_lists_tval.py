import numpy as np
import os

dir = "/scratch/gpfs/KNORMAN/rkempner/clustering_output/"
X_list = []
searchlight_list = []
for index,file in enumerate(os.scandir(dir + "searchlights_collapseTR_just_means/")):
    file_name = file.name
    if ".csv" in file_name:
        if "NOT480" in file_name:
            print("Error: NOT480 ", file_name)
        if index % 10000 == 0:
            print(index)
        light_id = file_name.split(".")[0]
        features = np.genfromtxt(dir + "searchlights_collapseTR_just_means/" + file_name, delimiter = ",").tolist()
        searchlight_list += [light_id]
        X_list += [features]

print("extracted for /searchlights/ directory")
X = np.vstack(X_list)
np.save(file = dir + "kmeans_assignments_just_means/master_X_list_just_means.npy", arr = X)
np.save(file = dir + "kmeans_assignments_just_means/master_searchlight_just_means.npy", arr = np.array(searchlight_list))

