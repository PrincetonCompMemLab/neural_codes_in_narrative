import json
import math
from scipy import stats
import msgpack
import orjson
from multiprocessing import Process
import numpy as np
import os


# this is to do the applied-to stuff2
# note that matching and applied-to are synonyms in my language
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
matching_id_labels_each_tr = np.array(matching_id_labels_each_tr)


apply_id_to_it = {2: matching_id_labels_each_tr == 2,
                  3: matching_id_labels_each_tr == 3,
                  4: matching_id_labels_each_tr == 4}

path_names = ['sameEv-sameSchema', 'sameEv-otherSchema', 
                            'otherEv-sameSchema', 'otherEv-otherSchema']

path_abbrev_dict = {"sEsS": 'sameEv-sameSchema',
                    "sEoS": 'sameEv-otherSchema',
                    "oEsS": 'otherEv-sameSchema',
                    "oEoS": 'otherEv-otherSchema'}

roi_to_focus = {"schema": ["2_2", "3_2", "4_2", "2_3", "3_3", "4_3", "2_4", "3_4", "4_4"],
                "path": ["2_2", "3_2", "4_2", "2_3", "3_3", "4_3", "2_4", "3_4", "4_4"],
                "rotated": ["2_3", "3_2"],
                "perception": ["2_2","3_3","4_4"]}



def get_perception_value_each_tr(path_to_trs):
    return (path_to_trs['sameEv-sameSchema'] + path_to_trs['sameEv-otherSchema']) \
           - (path_to_trs['otherEv-sameSchema'] + path_to_trs['otherEv-otherSchema'])
   

# 2_3 and 3_2
def get_rotated_value_each_tr(path_to_trs, focus):
    if focus == "2_3":
        return (path_to_trs['otherEv-sameSchema'] + path_to_trs['otherEv-otherSchema']) \
                - (path_to_trs['sameEv-sameSchema'] + path_to_trs['sameEv-otherSchema'])
    elif focus == "3_2":
        return (path_to_trs['otherEv-sameSchema'] + path_to_trs['sameEv-otherSchema']) \
                - (path_to_trs['sameEv-sameSchema'] + path_to_trs['otherEv-otherSchema'])


def get_path_value_each_tr(path_to_trs):
    return path_to_trs['sameEv-sameSchema'] - path_to_trs['otherEv-sameSchema']

def get_schema_value_each_tr(path_to_trs):
    return (path_to_trs['sameEv-sameSchema'] + path_to_trs['otherEv-sameSchema']) \
            - (path_to_trs['sameEv-otherSchema'] + path_to_trs['otherEv-otherSchema'])

# roi to event of the form x_y where x is the template event and y is the applied-to event
def get_roi_to_focus_to_measure(pid, wedding, template_to_pid_to_cond_to_matrices, roi_in):
    # cond for condition and path are synonyms
    focus_to_measure = {}
    this_measures_list = []
    for focus in roi_to_focus[roi_in]:
        template_id = focus[0]
        appliedto_id = focus[2]
        apply_it = apply_id_to_it[int(appliedto_id)]
        path_to_trs = {}
        for path in path_names:
            # get the trs for this wedding, and in the correct appliedto id
            new_trs = np.array(template_to_pid_to_cond_to_matrices[template_id][pid][path], dtype = float)[wedding,:][apply_it]
            path_to_trs[path] = new_trs
        if roi_in == "schema":
            new_measure = get_schema_value_each_tr(path_to_trs).tolist()
        elif roi_in == "perception":
            new_measure = get_perception_value_each_tr(path_to_trs).tolist()
        elif roi_in == "rotated":
            new_measure = get_rotated_value_each_tr(path_to_trs, focus).tolist()
        elif roi_in == "path":
            new_measure = get_path_value_each_tr(path_to_trs).tolist()
        else:
            print("Error: roi invalid")
            return
        focus_to_measure[focus] = new_measure
        [this_measures_list.append(x) for x in new_measure]
    return focus_to_measure, this_measures_list
            

def get_template_to_pid_to_cond_to_matrix_msgpack(light_id, in_dir = "/scratch/gpfs/rk1593/clustering_output/"):
    template_to_pid_to_cond_to_matrices = {}
    with open(in_dir + "searchlights_matrices_msgpack/" + light_id, "rb") as json_file:
        template_to_pid_to_cond_to_matrices = msgpack.load(json_file, strict_map_key = False) 
    return template_to_pid_to_cond_to_matrices

def get_template_to_pid_to_cond_to_matrix(light_id, in_dir = "/scratch/gpfs/rk1593/clustering_output/"):
    template_to_pid_to_cond_to_matrices = {}
    with open(in_dir + "searchlights_matrices_orjson/" + light_id, "rb") as f:
        template_to_pid_to_cond_to_matrices = orjson.loads(f.read())
    return template_to_pid_to_cond_to_matrices

def get_roi_to_pvalue(pid_to_roi_to_measure):
    roi_to_pvalue = {}
    for roi in roi_to_focus.keys():
        across_pids = []
        for pid in pid_to_roi_to_measure:
            across_pids.append(pid_to_roi_to_measure[pid][roi])
        tstat,pval = stats.ttest_1samp(across_pids, popmean= 0, alternative = "greater")
        roi_to_pvalue[roi] = float(pval)
    return roi_to_pvalue

def get_event_to_measure(focus_to_measure, roi):
    event_list = [2,3,4] if roi != "rotated" else [2,3]
    event_to_measure = {}
    for event in event_list:
        if roi == "schema" or roi == "path":
            avg_measure = np.nanmean(np.hstack((focus_to_measure["2_" + str(event)],
                        focus_to_measure["3_" + str(event)],
                        focus_to_measure["4_" + str(event)])))
        elif roi == "perception":
            avg_measure = np.nanmean(focus_to_measure[str(event) + "_" + str(event)])
        elif roi == "rotated":
            if event == 3:
                avg_measure = np.nanmean(focus_to_measure["2_3"])
            elif event == 2:
                avg_measure = np.nanmean(focus_to_measure["3_2"])
        event_to_measure[str(event)] = float(avg_measure)
    return event_to_measure

def process_chunk(light_list, output_dir):
    for light_id in light_list:
        template_to_pid_to_cond_to_matrices = get_template_to_pid_to_cond_to_matrix(light_id)
        pid_to_roi_to_measure = {}
        pid_to_roi_to_wedding_to_event_to_measure = {}
        for pid in template_to_pid_to_cond_to_matrices["2"]:
            if pid not in pid_to_roi_to_measure:
                pid_to_roi_to_measure[pid] = {}
                pid_to_roi_to_wedding_to_event_to_measure[pid] = {}
            for roi in roi_to_focus.keys():
                if roi not in pid_to_roi_to_wedding_to_event_to_measure[pid]:
                    pid_to_roi_to_wedding_to_event_to_measure[pid][roi] = {}
                measures_list = []
                for wedding in range(12):
                    # focus
                    focus_to_measure, new_measures = get_roi_to_focus_to_measure(pid, wedding,
                                            template_to_pid_to_cond_to_matrices, roi)
                    pid_to_roi_to_wedding_to_event_to_measure[pid][roi][str(wedding)] = get_event_to_measure(focus_to_measure, roi)
                    measures_list += new_measures
                # this takes average across weddings and tr's and focuses
                pid_to_roi_to_measure[pid][roi] = np.nanmean(measures_list)
        # "for this searchlight and this measure, is the measure reliably above zero across subjects
        roi_to_pval = get_roi_to_pvalue(pid_to_roi_to_measure)
        # light pvals
        if not os.path.exists(output_dir + light_id + "_RoiToPval"):
            with open(output_dir + light_id + "_RoiToPval", "wb") as f:
                f.write(orjson.dumps(roi_to_pval))
        # pid wedding event level neural measures
        neural_measures_path = "/scratch/gpfs/rk1593/clustering_output/searchlights_distilled_neural_measures/"
        if not os.path.exists(neural_measures_path + "each_searchlight/" + light_id):
            with open(neural_measures_path + "each_searchlight/" + light_id, "wb") as f:
                f.write(orjson.dumps(pid_to_roi_to_wedding_to_event_to_measure))

        # the way you can know if doing averages over chunk averages works 
        # as the same as aggregating than averaging is knowing
        # whether or not each chunk is the same size

def divide_chunks(l, chunk_size):
    # looping till length l
    for i in range(0, len(l), chunk_size): 
        yield l[i:i + chunk_size]

# event should be of the form x_y where x is the template event and y is the applied-to event
job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
output_dir = "/scratch/gpfs/rk1593/clustering_output/brainmap_pvals/each_searchlight/"
json_path = "/scratch/gpfs/rk1593/clustering_output/jobs_info_dict_manual_jupyter_without_tuples.json"
f = open(json_path,)
jobs_info_dict = json.load(f)
this_job_searchlights = jobs_info_dict["job_id_to_searchlight_subset"][str(job_id)]
num_chunks = 2
chunk_size = math.floor(len(this_job_searchlights) / num_chunks)
chunks_of_searchlights = divide_chunks(this_job_searchlights, chunk_size = chunk_size)
processes_list = []

for index,chunk in enumerate(chunks_of_searchlights):
    new_p = Process(target = process_chunk, args = (chunk, output_dir))
    new_p.start()
    processes_list.append(new_p)

for p in processes_list:
    p.join()