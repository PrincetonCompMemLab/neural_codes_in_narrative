import numpy as np
import os

dir = "/scratch/gpfs/rk1593/clustering_output/"
X_list = []
searchlight_list = []
for index,file in enumerate(os.scandir(dir + "searchlights_tval/")):
    file_name = file.name
    if ".csv" in file_name:
        if "NOT480" in file_name:
            print("Error: NOT480 ", file_name)
        if index % 10000 == 0:
            print(index)
        light_id = file_name.split(".")[0]
        features = np.genfromtxt(dir + "searchlights_tval/" + file_name, delimiter = ",").tolist()
        searchlight_list += [light_id]
        X_list += [features]

print("extracted for /searchlights/ directory")
X = np.vstack(X_list)
np.save(file = dir + "kmeans_assignments_tval/master_X_list_tval.npy", arr = X)
np.save(file = dir + "kmeans_assignments_tval/master_searchlight_tval.npy", arr = np.array(searchlight_list))

