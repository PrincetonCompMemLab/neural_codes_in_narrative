from logging import raiseExceptions
import os
import pdb
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from statistics import stdev
import json
from monkeylearn import MonkeyLearn
import math
import pandas as pd
from statesegmentation import GSBS
import matplotlib.pyplot as plt
import brainiak.eventseg.event
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import seaborn as sns 
import re
from nltk.stem import PorterStemmer # for finding the root words
from scipy.stats import wasserstein_distance, pearsonr, spearmanr
from plotnine import ggplot, geom_label, geom_point, aes, geom_text, stat_smooth, facet_wrap, geom_boxplot, geom_histogram, xlim, ylim, facet_grid

id_to_event_category_dict = {0:"background", 1:"wedding_start", 2: "celebrate_campfire",
                        3: "plant_flower", 4: "drop_coins", 5:"hold_torch", 6: "break_egg",
                        7:"draw_painting", 8:"receive_gifts"}

stemmer =  PorterStemmer()


stop_words = ['i','me','my','myself','we','our','you',"they're",'he', 'him', 'himself','she', "she'", 'her', "it'", 'it', "its", 'itself',
 'they','them','their','there','what','which','who','whom','this',"an", "also", "like", "not","ok", "okay", "oh", "would",
 'that','these','those','am','is','are',"was",'were','be','been',
 'have',"has", 'had','do','did','a','an', "not", "im", "sure", "when", "also", "I'm", "together", "couple", "wedding", "ceremony", "ritual",
 'the','and','but','if','or','because','as','of','at','by','for','to', "yet",
 'then','yeah', "Jolene", "Jeffrey", "Beth", "Jim", "Bob", "Jennifer", "Steph", "Harold",
 "Hannah", "Bill", "Camille", "Felix", "Doreen", "Simon", "Chris", "Magdalene", "Shannon", "Tobias", 
 "Vera", "Solomon", "Dennis", "Vivian", "Angela", "Dave", "like", "the", "where", "both", "maybe"]
stop_words = [stemmer.stem(x) for x in stop_words]
stop_words.append("Magdalene")
stop_words.append("magdalene")

couple_int_to_wedding_id = {0:1,1:2,2:6,3:17,4:19,5:20,6:22,
                            7:23,8:28,9:29,10:34,11:38}

all_pids_list = [2,3,4,5,6,7,8,9,10,11,12,13, 14, 15, 17, 18, 19, 22, 23, 24, 25, 26,27, 28, 29, 30, 31, 32, 33, 34,35, 36,37,38 ,39,40 ,41 , 42, 43, 44]

bad_words = ["\n"]



# PERCEIVED EMBEDDINGS#

# REQUIRES: words in chronological order, sliding window size for # of words
# EFFECTS: creates a list of lists with the words 
def create_sliding_windows(words_in_time, window_size, input_dir, wedding_id, segment_id, yes_timestamps):
    if yes_timestamps:
        timestamped_text_path = input_dir + "timestamped_text"
        file_query = str(wedding_id) + "_" + str(segment_id)
        list_of_files = sorted(os.listdir(input_dir + "timestamped_text"))
        timestamped_file_name = ""
        for file in list_of_files:
            if file_query in file:
                timestamped_file_name = file
        if timestamped_file_name == "":
            print("Error: timestamped file not found.")
            exit(0)
    # the sliding windows are of the format [(["Hello my name is Bob"], time), (["my name is Bob Ross", time)]
    splitted_words = words_in_time.split()
    list_of_sliding_windows = []
    # get words to grab before and after
    before_after_grab = math.floor(window_size / 2)
    not_match_list = []
    for index,word in enumerate(splitted_words):
        if yes_timestamps:
            # get the time stamp of this word by using the index
            df = pd.read_csv(input_dir + "timestamped_text/" + timestamped_file_name, header = None)
            time_stamp = df[2][index]
            if word != df[0][index]:
                not_match_list.append((word,df[0][index]))
                print(word, " != ", df[0][index])
        else:
            time_stamp = -1
        # TODO handle missing times
        # # if the time stamp is nan, get the average of the previous and next
        # if np.isnan(time_stamp):
        #     # if we have a word before and after
        #     if (index + 1) < len(df) and index >= 1:
        #         time_stamp = (df[2][index + 1] + df[3][index - 1]) / 2
        #     # if don't have a word after but have one befpre
        #     elif (index + 1) >= len(df) and index >= 1:
        #         time_stamp = df[2][index - 1]
        # if file_query == "1_5" and word == "of":
        #     pdb.set_trace()
        # check whether the words match or not

        # now go through the before and after grab
        this_window = []
        # before grab
        for i in range(index - before_after_grab, index):
            if i >= 0:
                this_window.append(stemmer.stem(splitted_words[i]))

        # put center word in
        # if stemmer.stem(word).lower() == "magdalene":
        #     pdb.set_trace()
        this_window.append(stemmer.stem(word))
        # after grab
        for i in range(index + 1, index + before_after_grab + 1):
            if i < len(splitted_words):
                this_window.append(stemmer.stem(splitted_words[i]))
        new_element = (" ".join(this_window), time_stamp)
        list_of_sliding_windows.append(new_element)
    return list_of_sliding_windows

# REQUIRES: directory of text files of the annotated clips, input_dir 
# is relative to the place where the file is ran
# EFFECTS: creates bag of words of all of the annotated words, and removes 
# punctuatiion and stems possesive nouns
def create_bag_of_words_and_sliding_windows(input_dir, window_size = 10):
    # create a bag of all of the words
    one_bag_of_words = []
    # create a list where each element is all the words in the particular segement
    # create list of words not not be included
    # the sliding windows are of the format [(["Hello my name is Bob"], time), (["my name is Bob Ross", time)]
    wedding_id_to_segment_id_to_sliding_windows_dict = {}
    # go through all the files in the directory for the perceived text
    list_of_names = sorted(os.listdir(input_dir + "raw_wedding_text"))
    #pdb.set_trace
    for file_name in list_of_names:
        if "txt" in file_name:
            # get the id of the wedding
            wedding_id = int(file_name.split(".")[1].split("_")[0])
            segment_id = int(file_name.split(".")[1].split("_")[1])
           
            if wedding_id not in wedding_id_to_segment_id_to_sliding_windows_dict:
                wedding_id_to_segment_id_to_sliding_windows_dict[wedding_id] = {}
            with open(input_dir + "raw_wedding_text/" + file_name) as f:
                all_lines = f.readlines()
            # remove commas from all lines
            for i in range(0,len(all_lines)):
                all_lines[i] = all_lines[i].replace(",","")
                all_lines[i] = all_lines[i].replace("'","")
                all_lines[i] = all_lines[i].replace("!","")
                all_lines[i] = all_lines[i].replace("?","")
                all_lines[i] = all_lines[i].replace(".","")
                all_lines[i] = all_lines[i].replace(":","")
                all_lines[i] = all_lines[i].replace(";","")
                all_lines[i] = all_lines[i].replace("(","")
                all_lines[i] = all_lines[i].replace(")","")
                all_lines[i] = all_lines[i].replace("\"","")
                all_lines[i] = all_lines[i].replace("-","")
                all_lines[i] = all_lines[i].replace("--","")
            # retain only character and replace all others with space
            # TODO: screening of words like removing stop words, TODO fix thing like " turning into \xe2\"
            # get all words in one place for sliding window approach
            [[one_bag_of_words.append(stemmer.stem(word)) for word in line.split() if word not in bad_words] for line in all_lines]
            # get list of sliding window bags separated by segments
            for line in all_lines:
                if line == "\n":
                    continue
                # DEBUG
                # for word in line.split():
                #     if "90" == stemmer.stem(word):
                #         pass
                #         print(file_name)
                # then submit that line to a function which will output
                # a list with all the sliding windows
                list_of_sliding_windows = create_sliding_windows(line, 
                                window_size = window_size, input_dir = input_dir,
                                wedding_id = wedding_id,
                                segment_id = segment_id,
                                yes_timestamps= False)
                wedding_id_to_segment_id_to_sliding_windows_dict[wedding_id][segment_id] = list_of_sliding_windows
    return one_bag_of_words, wedding_id_to_segment_id_to_sliding_windows_dict
#one_bag_of_words, wedding_id_to_segment_id_to_sliding_windows_dict = create_bag_of_words_and_sliding_windows("../../stimuli/")

# REQUIRES: lda model and vectorizer, w1 and w2 is a string of wordds
def get_corr_between_two_windows(lda, vectorizer, w1, w2):
    w1 = " ".join([stemmer.stem(x) for x in w1.split()])
    w2 = " ".join([stemmer.stem(x) for x in w2.split()])

    # 1
    vectorized_window1 = vectorizer.transform([w1])
    topic_vector1 = lda.transform(vectorized_window1).flatten()
    # 2
    vectorized_window2 = vectorizer.transform([w2])
    topic_vector2 = lda.transform(vectorized_window2).flatten()
    # corr
    print(topic_vector1)
    print(topic_vector2)
    return pearsonr(topic_vector1, topic_vector2)

# REQUIRES: bag of words, list of window bags with words
# EFFECTS: adds all the words from the recall data, and sliding windows from each, ignoring the I am done, into the
# bag of words and list of window bags to add to the CountVectorizer vocabulary and use to train the model
def add_recall_words_and_sliding_windows(one_bag_of_words, list_of_window_bags, recall_window_size, input_dir = "../../data/"):
    pid_directories = [x[0] for x in os.walk(input_dir + "raw_transcribed/") if "pid" in x[0]]
    wedding_index = 1000
    for pid_dir in pid_directories:
        list_of_names = sorted(os.listdir(pid_dir))
        for file_name in list_of_names:
     
            if "txt" in file_name:
                wedding_index += 1
                # get the id of the wedding
                with open(pid_dir + "/" + file_name) as f:
                    all_lines = f.readlines()
                # remove commas from all lines
                for i in range(0,len(all_lines)):
                    all_lines[i] = all_lines[i].replace(",","")
                    all_lines[i] = all_lines[i].replace("'","")
                    all_lines[i] = all_lines[i].replace("!","")
                    all_lines[i] = all_lines[i].replace("?","")
                    all_lines[i] = all_lines[i].replace(".","")
                    all_lines[i] = all_lines[i].replace(":","")
                    all_lines[i] = all_lines[i].replace(";","")
                    all_lines[i] = all_lines[i].replace("(","")
                    all_lines[i] = all_lines[i].replace(")","")
                    all_lines[i] = all_lines[i].replace("\"","")
                    all_lines[i] = all_lines[i].replace("-","")
                    all_lines[i] = all_lines[i].replace("--","")
                all_lines = [all_lines[0].split("#")[0]]
                # retain only character and replace all others with space
                # TODO: screening of words like removing stop words, TODO fix thing like " turning into \xe2\"
                # get all words in one place for sliding window approach
                [[one_bag_of_words.append(stemmer.stem(word)) for word in line.split() if word not in bad_words] for line in all_lines]
                for line in all_lines:
                        if line == "\n":
                            continue

                        # DEBUG
                        # for word in line.split():
                        #     if "90" == stemmer.stem(word):
                        #         print(file_name)
                        
                        # then submit that line to a function which will output
                        # a list with all the sliding windows
                        list_of_sliding_windows = create_sliding_windows(line, 
                                        window_size = recall_window_size, input_dir = input_dir,
                                        wedding_id = -1 ,
                                        segment_id = -1, 
                                        yes_timestamps= False)
                        # add al the sliding windows from this recall
                        [list_of_window_bags.append(x[0]) for x in list_of_sliding_windows] 
                        # append more wedding indices for the hyper plot stuff
    return one_bag_of_words, list_of_window_bags

# REQUIRES: one bag of words, and then bags of each window
# EFFECTS: a number-of-windows by number-of-words word-count matrix and the vectorizer
def create_windows_by_num_words_word_count_matrix(bag_of_all_words, list_of_window_bags, recall_window_size):
    bag_of_all_words, list_of_window_bags = add_recall_words_and_sliding_windows(bag_of_all_words, list_of_window_bags, recall_window_size= recall_window_size)
    # turn bag of all words to unique words for the vocabulary
    vectorizer = CountVectorizer(stop_words = stop_words, token_pattern = r"(?u)\b\S+\b")
    vectorizer.fit(set(bag_of_all_words))
    # now go through eahc window and get a new vector for that window
    vectors_list = []
    # go through every single sliding window in both perceived and recall
    # and create embedding for it and turn that into an array to later
    # fit the lda model on 
    for i in range(len(list_of_window_bags)):
        new_vector = vectorizer.transform([list_of_window_bags[i]])
        vectors_list.append(new_vector.toarray())
    output_array = np.vstack(vectors_list)
    return output_array, vectorizer

# bag_of_all_words, list_of_window_bags = create_bag_of_words("../../../data/wedding_text/")
# create_windows_by_num_words_word_count_matrix(bag_of_all_words, list_of_window_bags)

# REQUIRES: list of bags of words (e.g. ["hello ross", "I am Bob"]) and corresponding labels
# EFFECTS: outputs the same lists in a different order so that same labels are together
def put_same_event_labels_together(word_bags_list, labels_list):

    zipped_lists = zip(labels_list ,word_bags_list)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_word_bags = [element for _, element in sorted_zipped_lists]
    sorted_labels_list = sorted(labels_list)
    return sorted_word_bags, sorted_labels_list

# REQUIRES: number of windows by number of words in vocabulary matrix
# EFFECTS: do LDA topic modeling
# https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
def train_and_getLDA_topic_model(perceived_window_size = 10, input_dir = "../../stimuli/", num_components = 10, recall_window_size = 4):
    #pdb.set_trace
    one_bag_of_words, wedding_id_to_segment_id_to_sliding_windows_dict = create_bag_of_words_and_sliding_windows(input_dir, window_size = perceived_window_size)
    # make a list of bags to train the topic model
    list_of_bags = []
    list_of_segment_id = []
    narrative_id = []
    for wedding_id in wedding_id_to_segment_id_to_sliding_windows_dict:
        for segment_id in wedding_id_to_segment_id_to_sliding_windows_dict[wedding_id]:
            for sliding_window,timestamp in wedding_id_to_segment_id_to_sliding_windows_dict[wedding_id][segment_id]:
                list_of_bags.append(sliding_window)
                list_of_segment_id.append(segment_id)
                narrative_id.append(wedding_id)
    # get word count matrix and vectorizer
    word_count_matrix, vectorizer = create_windows_by_num_words_word_count_matrix(one_bag_of_words, list_of_bags, recall_window_size)
    lda = LatentDirichletAllocation(n_components=  num_components)
    lda.fit(word_count_matrix)
    # get a dictionary where the sliding windows are replaced by their topic vectors
    wedding_id_to_segment_id_to_embedding_dict = {}
    #pdb.set_trace
    for wedding_id in wedding_id_to_segment_id_to_sliding_windows_dict:
        wedding_id_to_segment_id_to_embedding_dict[wedding_id] = {}
        for segment_id in wedding_id_to_segment_id_to_sliding_windows_dict[wedding_id]:
            list_of_topic_vectors_timestamps = []
            for sliding_window, timestamp in wedding_id_to_segment_id_to_sliding_windows_dict[wedding_id][segment_id]:
                vectorized_window = vectorizer.transform([sliding_window])
                topic_vector = lda.transform(vectorized_window).flatten()
                # if np.all(topic_vector == topic_vector[0]):
                #     topic_vector[0] += 1e-6
                #     topic_vector[1] -= 1e-6
                list_of_topic_vectors_timestamps.append((topic_vector, timestamp))
            wedding_id_to_segment_id_to_embedding_dict[wedding_id][segment_id] = list_of_topic_vectors_timestamps
    return lda, vectorizer, word_count_matrix, wedding_id_to_segment_id_to_sliding_windows_dict, wedding_id_to_segment_id_to_embedding_dict, list_of_bags, list_of_segment_id, narrative_id
# train_and_getLDA_topic_model()

def get_corr_between_two_windows(lda, vectorizer, w1, w2):
    # 1
    vectorized_window1 = vectorizer.transform([w1])
    topic_vector1 = lda.transform(vectorized_window1).flatten()
    # 2
    vectorized_window2 = vectorizer.transform([w2])
    topic_vector2 = lda.transform(vectorized_window2).flatten()
    # corr
    return pearsonr(topic_vector1, topic_vector2)

# RECALL #

# REQUIRES: input_dir
# window_size
# participant_id
# EFFECTS: gets wedding_id_to_sliding_windows_dict
def get_participant_sliding_windows(input_dir, window_size, participant_id):
    wedding_id_to_sliding_windows_dict = {}
    wedding_id_to_order_sequence_dict = {}
    # go through all the files in the directory
    file_path_dir = input_dir + "raw_transcribed/" + "pid_" + str(participant_id) + "/"
    list_of_names = sorted(os.listdir(file_path_dir))
    for file_name in list_of_names:
        if "txt" in file_name:
            # get the wedding ID
            wedding_id = couple_int_to_wedding_id[int(file_name.split(".")[1])]

            # first get the viewing number
            # read in the csv mapping viewing order to wedding id
            df = pd.read_csv("../../stimuli/participant_viewing_sequence/subj" + str(participant_id) + "_day2_viewing.csv")
            wedding_video_files = [x for x in df["stimFile1a"]]
            viewing_number = -99
            for index, file in enumerate(wedding_video_files):
                this_file_wedding_id = int(file.split("-")[1].split(".")[0])
                if this_file_wedding_id == wedding_id:
                    viewing_number = index
                    break
            if viewing_number == -99:
                print("Error: wedding id not found in viewing file")
                exit(0)

           
            # get the order sequence 
            order_sequence = [0,1]
            first_aORb = df["stimFile2"][viewing_number].split(".")[1][1]
            order_sequence.append(2) if first_aORb == "a" else order_sequence.append(3)
            second_aORb = df["stimFile3"][viewing_number].split(".")[1][1]
            order_sequence.append(4) if second_aORb == "a" else order_sequence.append(5)
            third_aORb = df["stimFile4"][viewing_number].split(".")[1][1]
            order_sequence.append(6) if third_aORb == "a" else order_sequence.append(7)
            order_sequence.append(8)
            wedding_id_to_order_sequence_dict[wedding_id] = order_sequence
            x = 1
            if wedding_id not in wedding_id_to_sliding_windows_dict:
                wedding_id_to_sliding_windows_dict[wedding_id] = {}
            with open(file_path_dir + file_name) as f:
                all_lines = f.readlines()
            # ignore the I am done, by use of the hashtag
            all_lines = [all_lines[0].split("#")[0]]
            # remove all the nonsense
            for i in range(0,len(all_lines)):
                all_lines[i] = all_lines[i].replace(",","")
                all_lines[i] = all_lines[i].replace("'","")
                all_lines[i] = all_lines[i].replace("!","")
                all_lines[i] = all_lines[i].replace("?","")
                all_lines[i] = all_lines[i].replace(".","")
                all_lines[i] = all_lines[i].replace(":","")
                all_lines[i] = all_lines[i].replace(";","")
                all_lines[i] = all_lines[i].replace("(","")
                all_lines[i] = all_lines[i].replace(")","")
                all_lines[i] = all_lines[i].replace("\"","")
                all_lines[i] = all_lines[i].replace("-"," ")
                all_lines[i] = all_lines[i].replace("--","")
                # the sliding windows are of the format [(["Hello my name is Bob"], time), (["my name is Bob Ross", time)]
            
            splitted_words = all_lines[0].split()
            list_of_sliding_windows = []
            # get words to grab before and after
            before_after_grab = math.floor(window_size / 2)
            for index,word in enumerate(splitted_words):

                this_window = []
                # before grab
                for i in range(index - before_after_grab, index):
                    if i >= 0:
                        this_window.append(stemmer.stem(splitted_words[i]))

                # put center word in
                # if stemmer.stem(word).lower() == "magdalen":
                #     pdb.set_trace()
                this_window.append(stemmer.stem(word))
                # after grab
                for i in range(index + 1, index + before_after_grab + 1):
                    if i < len(splitted_words):
                        this_window.append(stemmer.stem(splitted_words[i]))
                new_element = (" ".join(this_window), -1)
                list_of_sliding_windows.append(new_element)
            wedding_id_to_sliding_windows_dict[wedding_id] = list_of_sliding_windows
    return wedding_id_to_sliding_windows_dict, wedding_id_to_order_sequence_dict
#get_participant_sliding_windows(input_dir = "../../data/", window_size = 7, participant_id = 30)


# REQUIRES: participant_id denotes which participant we want the topic-proportions matrix,
#  input_dir denotes where we can get the data
#  window size denotes how to do the sliding window which needs to be the same as the trained model
# lda model that has been trained 
def get_participants_recall_topic_proportions(participant_id, window_size, lda_model, vectorizer, input_dir = "../../data/"):
    wedding_id_to_sliding_windows_dict, wedding_id_to_order = get_participant_sliding_windows(input_dir = input_dir, window_size = window_size, participant_id = participant_id)
    wedding_id_to_embedding_dict = {}
    #pdb.set_trace
    for wedding_id in wedding_id_to_sliding_windows_dict:
        wedding_id_to_embedding_dict[wedding_id] = {}
        list_of_topic_vectors_timestamps = []
        for sliding_window, timestamp in wedding_id_to_sliding_windows_dict[wedding_id]:
            vectorized_window = vectorizer.transform([sliding_window])
            topic_vector = lda_model.transform(vectorized_window).flatten()
        
            # if we get one of those nasty all stop words sliding windows
            # create a little noise to get a non-constant vector
            if np.all(topic_vector == topic_vector[0]):
                topic_vector[0] += 1e-6
                topic_vector[1] -= 1e-6

            list_of_topic_vectors_timestamps.append((topic_vector,timestamp))
        wedding_id_to_embedding_dict[wedding_id]= list_of_topic_vectors_timestamps
    return wedding_id_to_embedding_dict, wedding_id_to_sliding_windows_dict, wedding_id_to_order
#get_participants_topic_proportions(participant_id= 30, window_size=7, lda_model= None, vectorizer= None, input_dir = "../../data/")

# REQUIRES: wedding_id, particular order of sequence, wedding_id_to_segment_id_to_embeddings_dict
# EFFECTS: extracts from wedding_id_to_segment_id_to_embeddings_dict all of the embeddings in time
def get_participant_perceived_wedding_sequence(wedding_id, order_sequence, wedding_to_segment_to_embed_dict, wedding_to_segment_to_windows):
    list_of_embeds = []
    list_of_segment_id = []
    list_of_time_stamps = []
    list_of_windows = []
    for segment_id in order_sequence:
        for index,embedding in enumerate(wedding_to_segment_to_embed_dict[wedding_id][segment_id]):
            list_of_embeds.append(embedding[0])
            list_of_segment_id.append(segment_id)
            list_of_time_stamps.append(embedding[1])
            list_of_windows.append(wedding_to_segment_to_windows[wedding_id][segment_id][index])
    window_by_embedding_matrix = np.vstack(list_of_embeds)
    return window_by_embedding_matrix, list_of_segment_id, list_of_time_stamps, list_of_windows

# EFFECTS: gets a recall window by embeddings matrix for one participant and one wedding
# and a perceived window by embeddings matrix, and the third thing this function
# returns is all of the recall wedding to embeddings
def get_embeddings_matrices_for_participant_and_wedding_id(participant_id, 
    wedding_id, recall_window_size, 
    lda_model, vectorizer, 
    wedding_to_segment_to_sliding_windows,wedding_to_segment_to_embeddings, 
    input_dir = "../../data/"):
    # get the recall matrix
    recall_wedding_to_embeddings, recall_wedding_to_sliding_windows, wedding_to_order = get_participants_recall_topic_proportions(participant_id= participant_id, 
                    window_size= recall_window_size, 
                    lda_model= lda_model, 
                    vectorizer= vectorizer, input_dir = input_dir)
    recall_embeddings_for_this_wedding = recall_wedding_to_embeddings[wedding_id]
    # get the perceived matrix
    perceived_embedding_matrix, list_of_segment_id, list_of_time_stamps, list_of_windows = get_participant_perceived_wedding_sequence(wedding_id,
                     wedding_to_order[wedding_id], wedding_to_segment_to_embeddings, wedding_to_segment_to_sliding_windows)
    return recall_embeddings_for_this_wedding, perceived_embedding_matrix, recall_wedding_to_embeddings, recall_wedding_to_sliding_windows, wedding_to_order

# EFFECTS: get wedding to segment to event to sliding windows
def get_event_to_sliding_windows(wedding_to_segment_to_event_topic_vectors, 
                                    wedding_id_to_segment_id_to_sliding_windows_dict):
    wedding_to_segment_to_event_to_sliding_windows = {}
    for wedding_id in wedding_to_segment_to_event_topic_vectors:
        wedding_to_segment_to_event_to_sliding_windows[wedding_id] = {}
        for segment_id in wedding_to_segment_to_event_topic_vectors[wedding_id]:
            wedding_to_segment_to_event_to_sliding_windows[wedding_id][segment_id] = {}
            # get the bounds on the events for this wedding and segment
            bounds = wedding_to_segment_to_event_topic_vectors[wedding_id][segment_id]["bounds"]
            # traverse through each event in this wedding and segment
            for index,embedding in enumerate(wedding_to_segment_to_event_topic_vectors[wedding_id][segment_id]["embeddings"]):
                event_id = index
                # add all sliding windows in this event to dict
                this_wed_event_wndw = []
                lower_bound = bounds[index]
                lower_bound += 1 
                upper_bound = bounds[index + 1] + 1
                this_wed_event_wndw = [x[0] for x in wedding_id_to_segment_id_to_sliding_windows_dict[wedding_id][segment_id][lower_bound:upper_bound]]
                wedding_to_segment_to_event_to_sliding_windows[wedding_id][segment_id][event_id] = this_wed_event_wndw 
    return wedding_to_segment_to_event_to_sliding_windows

# EFFECTS: get matrix for every single segment of each wedding that the participant actually perceived,
# and outputs a list of (wedding, segment) tuples corresponding to each row in that matrix
def get_event_embeddings_for_all_perceived(wedding_to_segment_to_event_topic_vectors, 
                                    wedding_id_to_segment_id_to_sliding_windows_dict, 
                                    wedding_to_order_dict):
    embeddings_list = []
    wedding_segment_and_eventid_list = []
    wedding_to_segment_to_event_to_sliding_windows = {}
    for wedding_id in wedding_to_order_dict:
            wedding_to_segment_to_event_to_sliding_windows[wedding_id] = {}
            for segment_id in wedding_to_order_dict[wedding_id]:
                wedding_to_segment_to_event_to_sliding_windows[wedding_id][segment_id] = {}
                # get the bounds on the events for this wedding and segment
                bounds = wedding_to_segment_to_event_topic_vectors[wedding_id][segment_id]["bounds"]
                # traverse through each event in this wedding and segment
                for index,embedding in enumerate(wedding_to_segment_to_event_topic_vectors[wedding_id][segment_id]["embeddings"]):
                    event_id = index
                    # add all sliding windows in this event to dict
                    this_wed_event_wndw = []
                    lower_bound = bounds[index]
                    lower_bound += 1 
                    upper_bound = bounds[index + 1] + 1
                    this_wed_event_wndw = [x[0] for x in wedding_id_to_segment_id_to_sliding_windows_dict[wedding_id][segment_id][lower_bound:upper_bound]]
                    wedding_to_segment_to_event_to_sliding_windows[wedding_id][segment_id][event_id] = this_wed_event_wndw 
                    # get the embedding
                    embeddings_list.append(embedding[0])
                    wedding_segment_and_eventid_list.append((wedding_id, segment_id, event_id))
    return np.vstack(embeddings_list), wedding_segment_and_eventid_list, wedding_to_segment_to_event_to_sliding_windows

# REQUIRES: all_perceived_embeddings_matrix, wedding_segment_and_eventid_list, wedding_id
# EFFECTS: extracts matrix of perceived events only in the correct wedding
def extract_only_matrix_of_correct_wedding(all_perceived_embeddings_matrix, wedding_segment_and_eventid_list, wedding_id):
    nrow, ncol = all_perceived_embeddings_matrix.shape
    embeddings_list = []
    new_wedding_segment_and_eventid_list = []
    # get only the embedding rows which are in the correct wedding and stack them
    for row in range(nrow):
        if wedding_segment_and_eventid_list[row][0] == wedding_id:
            embeddings_list.append(all_perceived_embeddings_matrix[row,:])
            new_wedding_segment_and_eventid_list.append(wedding_segment_and_eventid_list[row])
    return np.vstack(embeddings_list), new_wedding_segment_and_eventid_list

# EFFECTS: outputs dictionary mapping wedding_id to correlation matrix, and 
# different scores for each , and recall wedding to sliding windows, and 
# recall wedding to event to list of most matching wedding and segments
def align_perceived_and_recall_event_embeds(wedding_to_segment_to_event_topic_vectors, 
                                    wedding_id_to_segment_id_to_sliding_windows_dict, 
                                    wedding_to_order_dict, recall_wedding_to_event_topic_vectors,
                                     recall_wedding_to_sliding_windows, best_matches_threshold, only_correct_perceived = False, wedding_to_best_model = None):
    all_perceived_embeddings_matrix, wedding_segment_and_eventid_list, perceived_wedding_to_segment_to_event_to_sliding_windows = get_event_embeddings_for_all_perceived(wedding_to_segment_to_event_topic_vectors, 
                                    wedding_id_to_segment_id_to_sliding_windows_dict, 
                                    wedding_to_order_dict)
    wedding_to_aligned_corr_matrix = {}
    recall_wedding_to_event_to_sliding_windows = {} 
    recall_wed_to_event_to_best_matches = {}
    wedding_to_segment_event_tuples = {}
    for wedding_id in recall_wedding_to_event_topic_vectors:
       

        # if we only want to align with the correct perceived matrix
        if only_correct_perceived:
            perceived_matrix_to_align, new_wed_seg_event_list = extract_only_matrix_of_correct_wedding(all_perceived_embeddings_matrix, 
                                            wedding_segment_and_eventid_list, 
                                            wedding_id)
            wedding_to_segment_event_tuples[wedding_id] = new_wed_seg_event_list
        # if we are in the all perceived mode
        # then for each recall wedding we are aligning with all 
        # perceived weddings
        else:
            perceived_matrix_to_align = all_perceived_embeddings_matrix


        recall_wedding_to_event_to_sliding_windows[wedding_id] = {}
        recall_wed_to_event_to_best_matches[wedding_id] = {}
        # get the recall embedding matrix 
        recall_matrix,bounds = recall_get_event_segmented_embedding_matrix_and_bounds(recall_wedding_to_event_topic_vectors, wedding_id)
        #pdb.set_trace
        # concatenate the recall matrix with all the perceived embeddings
        concat_mat = np.concatenate((perceived_matrix_to_align, recall_matrix))
        # get a list of codings
        p_or_r_list = []
        nrow_perceived = perceived_matrix_to_align.shape[0]
        nrow_recall = recall_matrix.shape[0]
        [p_or_r_list.append("p") for i in range(perceived_matrix_to_align.shape[0])]
        [p_or_r_list.append("r") for i in range(recall_matrix.shape[0])]
        # get the correlations between perceived and recall but then subset that correlation matrix
        concat_corr = np.corrcoef(concat_mat)
        perceived_recall_corr = concat_corr[0:nrow_perceived, nrow_perceived: (nrow_perceived + nrow_recall)]
        wedding_to_aligned_corr_matrix[wedding_id] = perceived_recall_corr

        # create the recall to wedding to event sliding windows
        # and at same time create the best matches dict
        num_events_in_recall = len(bounds) - 1
        # check our work
        if num_events_in_recall != len(recall_wedding_to_event_topic_vectors[wedding_id]["embeddings"]):
            print("Error: num events in recall mismatch")
            Exception("Error: num events in recall mismatch")
        for recall_event_id in range(num_events_in_recall):
            # do sliding windows
            this_wed_event_wndw = []
            lower_bound = bounds[recall_event_id]
            lower_bound += 1 
            upper_bound = bounds[recall_event_id + 1] + 1
            this_wed_event_wndw = [x[0] for x in recall_wedding_to_sliding_windows[wedding_id][lower_bound:upper_bound]]
            recall_wedding_to_event_to_sliding_windows[wedding_id][recall_event_id] = this_wed_event_wndw
            # get best matches
            # first get the column of perceived recall corr matrix that is of this recall event id
            recall_event_to_perceived_column = perceived_recall_corr[:,recall_event_id]
            # get matches with higher than 0.5 correlation
            best_matches_indices = np.where(recall_event_to_perceived_column > best_matches_threshold)[0]
            recall_wed_to_event_to_best_matches[wedding_id][recall_event_id] = best_matches_indices
    
    # if we are only aligning with the correct wedding then output separate wedding segment event id corresponding indices list
    if only_correct_perceived:
        output_wed_seg_eventid_correspondence = wedding_to_segment_event_tuples
    else:
        output_wed_seg_eventid_correspondence = wedding_segment_and_eventid_list
        
    return wedding_to_aligned_corr_matrix, output_wed_seg_eventid_correspondence, perceived_wedding_to_segment_to_event_to_sliding_windows, recall_wedding_to_event_to_sliding_windows, recall_wed_to_event_to_best_matches


# REQUIRES: wedding_id_to_segment_id_to_embedding_dict of the topic vectors, id's
def get_topic_vector_matrix(wedding_id_to_segment_id_to_embedding_dict, wed_id, seg_id):
    embeds_list = []
    for em in wedding_id_to_segment_id_to_embedding_dict[wed_id][seg_id]:
        embeds_list.append(em[0])
    return np.vstack(embeds_list), len(embeds_list)



# EVENT SEGMENTATION #
def create_custom_step_var(scale):
    def custom_step_var(step):
        return scale * (0.98 ** (step - 1))
    return custom_step_var
    
def _default_var_schedule(step):
    return 4 * (0.98 ** (step - 1))

def hmm_fitting(embeddings, timestamps, step_var, penalty_parameter = 0, wedding_id = None, k_range = range(2,20), split_merge = True):
    wasserstein_distance_list = []
    fitted_models = []
    best_hmm = None
    embeddings_matrix = np.vstack(embeddings)
    # the below correlation matrix of all the embeddings will be used to acquire
    # the within and between state correlation distributions and then to compute the
    # wasserstein distance between them
    embeds_corr = np.corrcoef(embeddings_matrix)
    # if all the embeddings are the same then we let K = 1
    all_embeds_are_same = False  
    if np.all(embeds_corr[0] == embeds_corr):
        all_embeds_are_same = True
        hmm_sim = brainiak.eventseg.event.EventSegment(n_events =  1, step_var = step_var, split_merge = True)
        hmm_sim.fit(embeddings_matrix)
        event_pred_over_time = np.argmax(hmm_sim.segments_[0], axis = 1)
        wasserstein_distance_list.append(1)
        fitted_models.append(hmm_sim)
        best_hmm = hmm_sim
    # get the number of embeddings and cap the k_range at that
    num_rows, num_cols = embeds_corr.shape
    k_range = [x for x in k_range if x < num_rows]
    # boolean to keep track if all the 
    for K in k_range:
        # don't do any of this if all the embeds are the same
        if all_embeds_are_same:
            break
        hmm_sim = brainiak.eventseg.event.EventSegment(n_events = K, step_var = step_var, split_merge = True)
        hmm_sim.fit(embeddings_matrix)
        event_pred_over_time = np.argmax(hmm_sim.segments_[0], axis = 1)
        # if we get an event pred with all the same then make K = 1
        if np.all(event_pred_over_time[0] == event_pred_over_time):
            all_embeds_are_same = True
            hmm_sim = brainiak.eventseg.event.EventSegment(n_events = 1, step_var = step_var, split_merge = True)
            hmm_sim.fit(embeddings_matrix)
            event_pred_over_time = np.argmax(hmm_sim.segments_[0], axis = 1)
            wasserstein_distance_list.append(1)
            fitted_models.append(hmm_sim)
            best_hmm = hmm_sim
            break

        within_state_correlations = []
        between_state_correlations = []
        # explore the upper triangle of the embedding correlation
        for i in range(0,num_rows - 1):
            for j in range(i + 1,num_cols):
                row_event_id = event_pred_over_time[i]
                col_event_id = event_pred_over_time[j]
                # within state
                if row_event_id == col_event_id:
                    within_state_correlations.append(embeds_corr[i,j])
                else:
                    between_state_correlations.append(embeds_corr[i,j])
  
        within_state_correlations = [x for x in within_state_correlations if not np.isnan(x)]
        between_state_correlations = [x for x in between_state_correlations if not np.isnan(x)]
        if within_state_correlations == [] or between_state_correlations == []:
            #pdb.set_trace
            x = 1
        distance_with_penality = wasserstein_distance(within_state_correlations, between_state_correlations)
        distance_with_penality -= penalty_parameter * K
        wasserstein_distance_list.append(distance_with_penality)
        fitted_models.append(hmm_sim)

    # now get find the K which maximizes the wasserstein distance
    max_index = wasserstein_distance_list.index(max(wasserstein_distance_list))
    optimal_K = k_range[max_index] if not all_embeds_are_same else 1
    if not all_embeds_are_same:
        best_hmm = fitted_models[max_index]
    # bounds returns the upper bound index on the previous event
    bounds = np.where(np.diff(np.argmax(best_hmm.segments_[0], axis=1)))[0].tolist() if not all_embeds_are_same else []
    
    bounds.insert(0,-1)
    # np.argmax(wedding_to_best_hmm[wedding_id].segments_[0], axis=1)
    bounds.append(len(np.argmax(best_hmm.segments_[0], axis=1)) - 1)
    # get a list of hard event id predictions
    event_pred_over_time = np.argmax(best_hmm.segments_[0], axis = 1)
    # now get the topic vector for each event i = 1, ... K
    event_embedding_list = []
    for event_id in range(0,optimal_K):
        event_id_indices = np.where(event_pred_over_time == event_id)[0]
        # get the begining and end of the event timestamp
        first_timepoint = timestamps[event_id_indices[0]]
        end_timepoint = timestamps[event_id_indices[-1]]
        time_interval = [first_timepoint, end_timepoint]

        this_event_embeddings = [embed for index, embed in enumerate(embeddings) if index in event_id_indices]
        this_event_embeds_matrix = np.vstack(this_event_embeddings)
        average_this_event_embeds = np.mean(this_event_embeds_matrix, axis = 0)
        event_embedding_list.append((average_this_event_embeds, time_interval))
    return optimal_K, best_hmm, bounds, event_embedding_list, wasserstein_distance_list

def gsbs_fitting(embeddings, timestamps, wedding_id, segment_id, kmax):
    embeddings_matrix = np.vstack(embeddings)
    gsbs_sim = GSBS(x = embeddings_matrix, kmax = kmax)
    gsbs_sim.fit()
    event_pred_over_time = [(x - 1) for x in gsbs_sim.states]
    bounds = np.where(np.diff(gsbs_sim.states))[0].tolist() 
    bounds.insert(0,-1)
    bounds.append(len(gsbs_sim.states.tolist()) - 1)
    event_pred_over_time = np.array([(x - 1) for x in gsbs_sim.states])
    optimal_K = len(set(event_pred_over_time))
    # now get the topic vector for each event i = 1, ... K
    event_embedding_list = []
    for event_id in range(0,optimal_K):
        event_id_indices = np.where(event_pred_over_time == event_id)[0]
        # get the begining and end of the event timestamp
        first_timepoint = timestamps[event_id_indices[0]]
        end_timepoint = timestamps[event_id_indices[-1]]
        time_interval = [first_timepoint, end_timepoint]
        this_event_embeddings = [embed for index, embed in enumerate(embeddings) if index in event_id_indices]
        this_event_embeds_matrix = np.vstack(this_event_embeddings)
        average_this_event_embeds = np.mean(this_event_embeds_matrix, axis = 0)
        event_embedding_list.append((average_this_event_embeds, time_interval))
    return optimal_K, gsbs_sim, bounds, event_embedding_list, None


# REQUIRES: event_seg_type {'hmm', 'gsbs'}
def get_fitted_model_and_event_vectors(event_seg_type, embeddings, timestamps, step_var, penalty_parameter = 0,  wedding_id = None, segment_id = None, k_range = range(2,20), kmax = 50):
    print(wedding_id, segment_id)
    if event_seg_type != "gsbs" and event_seg_type != "hmm":
        print("Error: invalid event_seg_type")
        exit(0)
    if event_seg_type == "hmm":
        return hmm_fitting(embeddings = embeddings, timestamps= timestamps, step_var = step_var, penalty_parameter= penalty_parameter,
                        wedding_id= wedding_id, k_range = k_range)  
    elif event_seg_type == "gsbs":
        return gsbs_fitting(embeddings, timestamps,wedding_id, segment_id, kmax)





# REQUIRES: wedding_id_to_segment_id_to_embedding_dict
# EFFECTS: outputs a wedding id to segment to list of topic vector for
# each event by taking the average of all topic vectors across the event 
# boundaries
def get_perceived_event_topic_vectors(event_seg_type, wedding_to_segment_to_embeddings, step_var_scale, penalty_parameter, k_range, kmax = 20):
    #pdb.set_trace
    step_var = create_custom_step_var(step_var_scale)
    all_optimal_K_list = []
    wedding_to_segment_to_event_topic_vectors = {}
    wedding_to_segment_to_wasser_list = {}
    for wedding_id in wedding_to_segment_to_embeddings:
        wedding_to_segment_to_event_topic_vectors[wedding_id] = {}
        wedding_to_segment_to_wasser_list[wedding_id]  = {}
        for segment_id in wedding_to_segment_to_embeddings[wedding_id]:
            # get list of embeddings
            embeddings_list = [emb[0] for emb in wedding_to_segment_to_embeddings[wedding_id][segment_id]]
            timestamps = [emb[1] for emb in wedding_to_segment_to_embeddings[wedding_id][segment_id]]
            # for each segment, run an optimized HMM and get the boundaries
            optimal_K, best_model, bounds, event_embedding_list, wasser_list = get_fitted_model_and_event_vectors(event_seg_type, embeddings_list, timestamps, step_var = step_var, penalty_parameter = penalty_parameter, wedding_id = wedding_id, segment_id = segment_id, k_range = k_range, kmax = kmax)
            wedding_to_segment_to_event_topic_vectors[wedding_id][segment_id] = {"embeddings": event_embedding_list,
                                                                                "bounds": bounds}
            wedding_to_segment_to_wasser_list[wedding_id][segment_id] = wasser_list
            all_optimal_K_list.append(optimal_K)
    return wedding_to_segment_to_event_topic_vectors, all_optimal_K_list, wedding_to_segment_to_wasser_list

# REQUIRES: wedding_to_segment_to_event_topic_vectors, wedding id, segment id
# returns: embedding matrix for events in a particular wedding and segment, the bounds and num timepoints
def perceived_get_event_segmented_embedding_matrix_and_bounds(wedd_to_seg_to_event_embed, wed_id, seg_id):
    list_of_embeds = []
    for em in wedd_to_seg_to_event_embed[wed_id][seg_id]["embeddings"]:
        list_of_embeds.append(em[0]) # 0  index into the embedding
    return np.vstack(list_of_embeds), wedd_to_seg_to_event_embed[wed_id][seg_id]["bounds"]



# REQUIRES: wedding_id_to_embedding_dict, penalty_parameter is the penalty on model complexity for the HMM objective function
# step_var is the function given to the HMM, k_range is the k events to try out for the HMM
# EFFECTS: outputs a wedding id to list of topic vector for
# each event by taking the average of all topic vectors across the event 
# boundaries
def get_recall_event_topic_vectors(event_seg_type, wedding_to_embeddings, penalty_parameter, step_var, k_range = range(2,20)):
    all_optimal_K_list = []
    wedding_to_event_topic_vectors = {}
    wedding_to_wasser_list = {}
    wedding_to_best_model = {}
    for wedding_id in wedding_to_embeddings:
        wedding_to_event_topic_vectors[wedding_id] = {}
        wedding_to_wasser_list[wedding_id] = {}
        # get list of embeddings
        embeddings_list = [emb[0] for emb in wedding_to_embeddings[wedding_id]]
        timestamps = [emb[1] for emb in wedding_to_embeddings[wedding_id]]
        # for each segment, run an optimized HMM and get the boundaries
        optimal_K, best_model, bounds, event_embedding_list, wasserstein_distance_list = get_fitted_model_and_event_vectors(event_seg_type, embeddings_list, timestamps, step_var = step_var, penalty_parameter= penalty_parameter, wedding_id= wedding_id, k_range= k_range)
        wedding_to_event_topic_vectors[wedding_id] = {"embeddings": event_embedding_list,
                                                                            "bounds": bounds}
        wedding_to_wasser_list[wedding_id] = wasserstein_distance_list
        wedding_to_best_model[wedding_id] = best_model
        all_optimal_K_list.append(optimal_K)
    return wedding_to_event_topic_vectors, all_optimal_K_list, wedding_to_wasser_list, wedding_to_best_model

# REQUIRES: wedding_to_event_topic_vectors, wedding id, segment id
# returns: embedding matrix for events in a particular wedding and segment, the bounds and num timepoints
def recall_get_event_segmented_embedding_matrix_and_bounds(wedd_to_event_embed, wed_id):
    list_of_embeds = []
    for em in wedd_to_event_embed[wed_id]["embeddings"]:
        list_of_embeds.append(em[0]) # 0  index into the embedding
    return np.vstack(list_of_embeds), wedd_to_event_embed[wed_id]["bounds"]

# REQUIRES: wedding_id_to_segment_id_to_embedding_dict of the topic vectors, id's
def recall_get_topic_vector_matrix(wedding_id_to_embedding_dict, wed_id):

    embeds_list = []
    for em in wedding_id_to_embedding_dict[wed_id]:
        embeds_list.append(em[0])
    return np.vstack(embeds_list), len(embeds_list)


# PRECISION + DISTICTIVENESS #

# EFFECTS: precision score is the average precision for all recall events for all weddings
# note that this can be over all possible perceived weddding segments or just correct wedding segments
def get_participants_column_wise_precision_distinctiveness(wedding_to_aligned_corr_matrix):
    precision_list = []
    distinctiveness_list = []
    for wedding_id in wedding_to_aligned_corr_matrix:
        matrix = wedding_to_aligned_corr_matrix[wedding_id]
        nrows, ncols = matrix.shape
        for recall_event in range(ncols):
        
            #print(wedding_id, recall_event)
            this_col = matrix[:,recall_event]
            if not np.all(np.isnan(this_col)):
                high_match_index = np.where(this_col == np.max(this_col))
                this_precision = this_col[high_match_index]
                precision_list.append(this_precision[0])
                # get the standard deviation and mean of this column to z-score precision
                this_col_stdev = np.std(this_col)
                this_col_mean = np.mean(this_col)
                this_distinctiveness = (this_precision - this_col_mean) / this_col_stdev
                distinctiveness_list.append(this_distinctiveness[0])
    overall_precision = sum(precision_list) / len(precision_list)
    overall_distinctivenss = sum(distinctiveness_list) / len(distinctiveness_list)
    return {"manning_precision: ": overall_precision,
            "manning_distinctivness: ": overall_distinctivenss
           }

# REQUIRES: dicitonary mapping wedding id to the matrix containing 
# correlations between the perceived and recall events, for one participant
# as well as dictionary mapping the wedding to a list of the ritual id for each row of the matrix
# EFFECTS: for each row in each of these matrices, get the precision score
# and then output the average precision for a particular wedding via a dictionary
# and also output average precision over all rows in all weddings
def get_participants_row_wise_precision(wedding_to_aligned_corr_matrix, wedding_to_segment_event_ids, perceived_wedding_to_segment_to_event_to_sliding_windows, recall_wedding_to_event_to_sliding_windows, recall_wedding_to_event_topic_vectors, perceived_wedding_to_segment_to_event_topic_vectors):
    precision_list = []
    wedding_to_precision_dict = {}
    ritual_to_precisions_list = {}
    ritual_to_avg_precision_dict = {}
    wedding_to_ritual_to_avg_precision = {}
    #pdb.set_trace()
    # go through each wedding and get the precisions of the rows (perceived events)
    # in each 
    for wedding_id in wedding_to_aligned_corr_matrix:
        wedding_to_ritual_to_avg_precision[wedding_id] = {}
        ritual_id_labels = [x[1] for x in wedding_to_segment_event_ids[wedding_id]]
        matrix = wedding_to_aligned_corr_matrix[wedding_id]
        nrows, ncols = matrix.shape
        this_wedding_precision_list = []
        this_wedding_and_ritual_prec_list = []

        for perceived_event in range(nrows):
            ritual_id = ritual_id_labels[perceived_event]
            this_row = matrix[perceived_event,:]
            if not np.all(np.isnan(this_row)):
                high_match_index = np.where(this_row == np.max(this_row))
                this_precision = this_row[high_match_index]
                precision_list.append(this_precision[0])
                this_wedding_and_ritual_prec_list.append(this_precision[0])
                this_wedding_precision_list.append(this_precision[0])
                if ritual_id not in ritual_to_precisions_list:
                    ritual_to_precisions_list[ritual_id] = []
                ritual_to_precisions_list[ritual_id].append(this_precision[0])
            # get the avg precision for this ritual and reset list
            if perceived_event == (nrows - 1):
                if len(this_wedding_and_ritual_prec_list) == 0:
                    print("919")
                    wedding_to_ritual_to_avg_precision[wedding_id][ritual_id] = np.nan
                else:
                    previous_ritual_avg = sum(this_wedding_and_ritual_prec_list) / len(this_wedding_and_ritual_prec_list)
                    wedding_to_ritual_to_avg_precision[wedding_id][ritual_id] = previous_ritual_avg
            # if we are not done iterating through these rows and the ritual id is not equal to the next one,
            # then we are at the end of this ritual so we want to get the avg precision for those rows
            # and then rest the list
            elif ritual_id != ritual_id_labels[perceived_event + 1]:
                if len(this_wedding_and_ritual_prec_list) == 0:
                    print("929")
                    wedding_to_ritual_to_avg_precision[wedding_id][ritual_id] = np.nan
                else:
                    previous_ritual_avg = sum(this_wedding_and_ritual_prec_list) / len(this_wedding_and_ritual_prec_list)
                    wedding_to_ritual_to_avg_precision[wedding_id][ritual_id] = previous_ritual_avg
                this_wedding_and_ritual_prec_list = []
        # obtain average precision for this wedding
        this_wed_avg_p = sum(this_wedding_precision_list) / len(this_wedding_precision_list)
        wedding_to_precision_dict[wedding_id] = this_wed_avg_p
    # get avg precision in each ritual
    for ritual_id in ritual_to_precisions_list:
        ritual_to_avg_precision_dict[ritual_id] =  sum(ritual_to_precisions_list[ritual_id]) / len(ritual_to_precisions_list[ritual_id])
    overall_precision = sum(precision_list) / len(precision_list)
    return overall_precision, wedding_to_precision_dict, ritual_to_avg_precision_dict, wedding_to_ritual_to_avg_precision

# REQUIRES: all wedding ritual events ids perceived for a participant, and the wedding id
# EFFECTS: returns a list of ritual id's that are the ritual id's that the participant 
# received in this particular wedding 
def get_ritual_ids_in_perceived_wedding(all_perceived_wedding_ritual_event_ids, target_wedding_id):
    # below we go through all the wedding ritual event id labels
    # and we gather the ritual id's that are in our wedding of interest
    ritual_id_list = []
    for wedding_id, segment_id, event_id in all_perceived_wedding_ritual_event_ids:
        if wedding_id == target_wedding_id:
            if segment_id not in ritual_id_list:
                ritual_id_list.append(segment_id)
    return ritual_id_list



# REQUIRES: 
# (1) dicitonary mapping recall wedding id to matrix containing aligned correlations with all perceived events
# (2) we ignore the segment id's that is not in "ritual_types"
# (3) wedding_segment_event_id_list
def get_participants_in_ritual_distinctiveness(recall_wedding_to_aligned_corr_matrix, all_perceived_wedding_ritual_event_ids, recall_wedding_to_event_to_sliding_windows, perceived_wedding_to_segment_to_event_to_sliding_windows,  allowed_ritual_types = set([0,1,2,3,4,5,6,7,8]), debug = True):
    # these two lists maintain data for the other three below
    ritual_to_dvns_list = {}
    for rid in allowed_ritual_types: # initialize
        ritual_to_dvns_list[rid] = []
    wedding_to_dvns_list = {}
    overall_dvns_list = []
    # these three lists will be outputted
    ritual_to_avg_dvns = {}
    wedding_to_avg_dvns = {}
    recall_wedding_id_to_ritual_id_to_dvns = {} 
    if debug:
        overall_dvns = 0
        return overall_dvns, wedding_to_avg_dvns, ritual_to_avg_dvns, recall_wedding_id_to_ritual_id_to_dvns
    # go through each matrix for each recall wedding
    for recall_wedding_id in recall_wedding_to_aligned_corr_matrix:
        recall_wedding_id_to_ritual_id_to_dvns[recall_wedding_id] = {}
        wedding_to_dvns_list[recall_wedding_id] = []
        # get the aligned corr matrix for this wedding
        matrix = recall_wedding_to_aligned_corr_matrix[recall_wedding_id]
        nrows, ncols = matrix.shape
        # get the rituals in the correct perceived for this wedding
        rituals_in_correct_perceived = get_ritual_ids_in_perceived_wedding(all_perceived_wedding_ritual_event_ids, recall_wedding_id)

        # go through each perceived ritual in the intersection of the allowed ritual types
        for target_perceived_ritual in set(rituals_in_correct_perceived).intersection(allowed_ritual_types):
            #print(recall_wedding_id, target_perceived_ritual)
            # for each wedding that contains this ritual, get a list of the
            # precision value for all the rows/events in the ritual
            wedding_to_precisions_in_ritual_rows = {}
            
            # get the average precision for each wedding that has this ritual type
            # go through each row/event in the matrix
            for p_event_row in range(nrows):
                # if the ritual is the perceived ritual of interest
                p_wedding_id, p_ritual_id, p_event_id = all_perceived_wedding_ritual_event_ids[p_event_row]
                if p_ritual_id == target_perceived_ritual:
                    # set up the dict
                    if p_wedding_id not in wedding_to_precisions_in_ritual_rows:
                        wedding_to_precisions_in_ritual_rows[p_wedding_id] = []
                    # get the precision value of this row
                    this_row = matrix[p_event_row,:]
                    if not np.all(np.isnan(this_row)):
                        high_match_index = np.where(this_row == np.max(this_row))
                        this_precision = this_row[high_match_index]
                        wedding_to_precisions_in_ritual_rows[p_wedding_id].append(this_precision[0])
            wedding_to_avg_precision_in_ritual = {}
           
            for w_id in wedding_to_precisions_in_ritual_rows:
                avg_precision = sum(wedding_to_precisions_in_ritual_rows[w_id]) / len(wedding_to_precisions_in_ritual_rows[w_id])
                #avg_precision = max(wedding_to_precisions_in_ritual_rows[w_id])
                wedding_to_avg_precision_in_ritual[w_id] = avg_precision
     
            # now get the distinctiveness for this ritual in this wedding
            
            if recall_wedding_id in wedding_to_avg_precision_in_ritual:
                correct_wedding_avg_precision = wedding_to_avg_precision_in_ritual[recall_wedding_id]
                mean_avg_precisions = sum(list(wedding_to_avg_precision_in_ritual.values())) / len(list(wedding_to_avg_precision_in_ritual.values()))
                stdev_avg_precisions = stdev(list(wedding_to_avg_precision_in_ritual.values()))
                dvns_this_wedding_ritual = (correct_wedding_avg_precision - mean_avg_precisions) / stdev_avg_precisions
                overall_dvns_list.append(dvns_this_wedding_ritual)
                wedding_to_dvns_list[recall_wedding_id].append(dvns_this_wedding_ritual)
                ritual_to_dvns_list[target_perceived_ritual].append(dvns_this_wedding_ritual)
            else:
                print("1024")
                dvns_this_wedding_ritual = np.nan
            recall_wedding_id_to_ritual_id_to_dvns[recall_wedding_id][target_perceived_ritual] = dvns_this_wedding_ritual
    # now populate our output lists
    for ritual_id in ritual_to_dvns_list:
        ritual_to_avg_dvns[ritual_id] = sum(ritual_to_dvns_list[ritual_id]) / len(ritual_to_dvns_list[ritual_id])
    for wedding_id in wedding_to_dvns_list:
        wedding_to_avg_dvns[wedding_id] = sum(wedding_to_dvns_list[wedding_id]) / len(wedding_to_dvns_list[wedding_id])
    overall_dvns = sum(overall_dvns_list) / len(overall_dvns_list)
    return overall_dvns, wedding_to_avg_dvns, ritual_to_avg_dvns, recall_wedding_id_to_ritual_id_to_dvns

# REQUIRES: json path, dict
def dump_dict_to_json(json_file_path, new_dict):
    json_string = json.dumps(new_dict)
    json_file = open(json_file_path, "w")
    json_file.write(json_string)
    json_file.close()


# REQUIRES: recall window size, lda model, vectorizer, perceived wedding_to_segment_to_sliding_windows dict,
# wedding_to_segment_to_event_topic_vectors perceived created by optimal HMM search on perceived 
# perceived wedding_to_segment_to_embeddings dict, list of participant ids
# EFFECTS: dictionary mapping participant to their precision and dictinctivness stuff
def get_all_participants_metrics(event_seg_type, lda_model, 
                    vectorizer, perceived_wedding_to_segment_to_sliding_windows,
                        perceived_wedding_to_segment_to_embeddings,
                        perceived_wedding_to_segment_to_event_topic_vectors,
                        p_ids, recall_window_size, step_var_scale, penalty_parameter, k_range, try_and_except = True, pre_saved_dict = None):
    new_saved_dict = {}
    participant_to_metrics = {}
    participant_to_other_data = {}
    all_participants_optimal_K_list = []
    step_var = create_custom_step_var(step_var_scale)
    for p_id in p_ids:
        print("--------")
        print("p_id: ",p_id)
        #pdb.set_trace
        if try_and_except:

            try:
                if pre_saved_dict == None:
                    recall_embeddings_for_this_wedding, perceived_embedding_matrix, recall_wedding_to_embeddings, recall_wedding_to_sliding_windows, wedding_to_order = get_embeddings_matrices_for_participant_and_wedding_id(participant_id = p_id, 
                                    wedding_id = 1, recall_window_size = recall_window_size, 
                                    lda_model = lda_model, vectorizer = vectorizer, 
                                    wedding_to_segment_to_sliding_windows = perceived_wedding_to_segment_to_sliding_windows,
                                    wedding_to_segment_to_embeddings = perceived_wedding_to_segment_to_embeddings)

                    recall_wedding_to_event_topic_vectors, all_optimal_K_list, wedding_to_wasser_list, wedding_to_best_model = get_recall_event_topic_vectors(event_seg_type, recall_wedding_to_embeddings, step_var = step_var, penalty_parameter= penalty_parameter, k_range = k_range)
                    print("got participants event seg")
                    [all_participants_optimal_K_list.append(x) for x in all_optimal_K_list]
                    this_pid_new_saved_dict = {
                                    "recall_wedding_to_embeddings":recall_wedding_to_embeddings,
                                    "recall_wedding_to_sliding_windows": recall_wedding_to_sliding_windows,
                                    "wedding_to_order": wedding_to_order,
                                    "recall_wedding_to_event_topic_vectors": recall_wedding_to_event_topic_vectors,
                                    "all_optimal_K_list": all_optimal_K_list,
                                    "wedding_to_wasser_list": wedding_to_wasser_list,
                                    "wedding_to_best_model": wedding_to_best_model
                                    }
                    new_saved_dict[p_id] = this_pid_new_saved_dict 
                    # json_file_path = "../../data/unfinished_business/participant_to_metrics/data_pid_" + str(p_id) + ".json"
                    # dump_dict_to_json(json_file_path, this_pid_new_saved_dict)
                else:
                
                    recall_wedding_to_embeddings = pre_saved_dict[p_id]["recall_wedding_to_embeddings"]
                    recall_wedding_to_sliding_windows = pre_saved_dict[p_id]["recall_wedding_to_sliding_windows"]
                    wedding_to_order = pre_saved_dict[p_id]["wedding_to_order"]
                    recall_wedding_to_event_topic_vectors = pre_saved_dict[p_id]["recall_wedding_to_event_topic_vectors"]
                    all_optimal_K_list = pre_saved_dict[p_id]["all_optimal_K_list"]

                # get precision
                p_r_aligned_only_correct_y_axis,wedding_to_segment_and_eventid_dict, perceived_wedding_to_segment_to_event_to_sliding_windows, recall_wedding_to_event_to_sliding_windows, recall_wed_to_event_to_best_matches = align_perceived_and_recall_event_embeds(
                                                perceived_wedding_to_segment_to_event_topic_vectors, 
                                                perceived_wedding_to_segment_to_sliding_windows, 
                                                wedding_to_order, recall_wedding_to_event_topic_vectors,
                                                                recall_wedding_to_sliding_windows,
                                                                best_matches_threshold = 0.5,
                                                                only_correct_perceived = True,
                                                                wedding_to_best_model = wedding_to_best_model)
                if p_id == 3:
                    sns.heatmap(pd.DataFrame(p_r_aligned_only_correct_y_axis[34]), vmin=0, vmax=1).set_title("Participant 3, Recall 34")
                overall_precision, wedding_to_precision_dict, ritual_to_avg_precision_dict, wedding_to_ritual_to_avg_precision = get_participants_row_wise_precision(p_r_aligned_only_correct_y_axis, wedding_to_segment_and_eventid_dict, perceived_wedding_to_segment_to_event_to_sliding_windows, recall_wedding_to_event_to_sliding_windows, recall_wedding_to_event_topic_vectors, perceived_wedding_to_segment_to_event_topic_vectors)
                print("got precision")
                # get within-ritual distinctiveness 
                p_r_aligned_all_perceived_y_axis,all_perceived_wedding_to_segment_and_eventid_list, all_perceived_wedding_to_segment_to_event_to_sliding_windows, recall_wedding_to_event_to_sliding_windows, recall_wed_to_event_to_best_matches = align_perceived_and_recall_event_embeds(
                                        perceived_wedding_to_segment_to_event_topic_vectors, 
                                        perceived_wedding_to_segment_to_sliding_windows, 
                                        wedding_to_order, recall_wedding_to_event_topic_vectors,
                                                        recall_wedding_to_sliding_windows,
                                                        best_matches_threshold = 0.5,
                                                        only_correct_perceived = False)
                new_saved_dict[p_id]["p_r_aligned_all_perceived_y_axis"] = p_r_aligned_all_perceived_y_axis
                new_saved_dict[p_id]["all_perceived_wedding_to_segment_and_eventid_list"] = all_perceived_wedding_to_segment_and_eventid_list
                new_saved_dict[p_id]["all_perceived_wedding_to_segment_to_event_to_sliding_windows"] = all_perceived_wedding_to_segment_to_event_to_sliding_windows
                new_saved_dict[p_id]["recall_wedding_to_event_to_sliding_windows"] = recall_wedding_to_event_to_sliding_windows
                overall_dvns, wedding_to_avg_dvns, ritual_to_avg_dvns, recall_wedding_id_to_ritual_id_to_dvns = get_participants_in_ritual_distinctiveness(p_r_aligned_all_perceived_y_axis,all_perceived_wedding_to_segment_and_eventid_list, recall_wedding_to_event_to_sliding_windows, all_perceived_wedding_to_segment_to_event_to_sliding_windows)
                print("got dvns")
                new_dict_pid = {"overall_precision": overall_precision,
                                        "wedding_to_avg_precision":  wedding_to_precision_dict,
                                        "ritual_to_avg_precision": ritual_to_avg_precision_dict,
                                        "wedding_to_ritual_to_avg_precision": wedding_to_ritual_to_avg_precision,
                                        "overall_dvns": overall_dvns,
                                        "wedding_to_avg_dvns": wedding_to_avg_dvns,
                                        "ritual_to_avg_dvns":ritual_to_avg_dvns,
                                        "recall_wedding_id_to_ritual_id_to_dvns":recall_wedding_id_to_ritual_id_to_dvns}
                participant_to_metrics[p_id] = new_dict_pid
                # json_file_path = "../../data/unfinished_business/participant_to_metrics/metrics_pid_" + str(p_id) + ".json"
                # dump_dict_to_json(json_file_path, new_dict_pid)
                participant_to_other_data[p_id] = {
                                        ""
                }
            except:
                return participant_to_metrics, new_saved_dict, all_participants_optimal_K_list
        elif not try_and_except:
                if pre_saved_dict == None:
                    recall_embeddings_for_this_wedding, perceived_embedding_matrix, recall_wedding_to_embeddings, recall_wedding_to_sliding_windows, wedding_to_order = get_embeddings_matrices_for_participant_and_wedding_id(participant_id = p_id, 
                                    wedding_id = 1, recall_window_size = recall_window_size, 
                                    lda_model = lda_model, vectorizer = vectorizer, 
                                    wedding_to_segment_to_sliding_windows = perceived_wedding_to_segment_to_sliding_windows,
                                    wedding_to_segment_to_embeddings = perceived_wedding_to_segment_to_embeddings)

                    recall_wedding_to_event_topic_vectors, all_optimal_K_list, wedding_to_wasser_list, wedding_to_best_model = get_recall_event_topic_vectors(event_seg_type, recall_wedding_to_embeddings, step_var = step_var, penalty_parameter= penalty_parameter, k_range = k_range)
                    print("got participants event seg")
                    [all_participants_optimal_K_list.append(x) for x in all_optimal_K_list]
                    this_pid_new_saved_dict = {
                                    "recall_wedding_to_embeddings":recall_wedding_to_embeddings,
                                    "recall_wedding_to_sliding_windows": recall_wedding_to_sliding_windows,
                                    "wedding_to_order": wedding_to_order,
                                    "recall_wedding_to_event_topic_vectors": recall_wedding_to_event_topic_vectors,
                                    "all_optimal_K_list": all_optimal_K_list,
                                    "wedding_to_wasser_list": wedding_to_wasser_list,
                                    "wedding_to_best_model": wedding_to_best_model
                                    }
                    new_saved_dict[p_id] = this_pid_new_saved_dict 
                    # json_file_path = "../../data/unfinished_business/participant_to_metrics/data_pid_" + str(p_id) + ".json"
                    # dump_dict_to_json(json_file_path, this_pid_new_saved_dict)
                else:
                
                    recall_wedding_to_embeddings = pre_saved_dict[p_id]["recall_wedding_to_embeddings"]
                    recall_wedding_to_sliding_windows = pre_saved_dict[p_id]["recall_wedding_to_sliding_windows"]
                    wedding_to_order = pre_saved_dict[p_id]["wedding_to_order"]
                    recall_wedding_to_event_topic_vectors = pre_saved_dict[p_id]["recall_wedding_to_event_topic_vectors"]
                    all_optimal_K_list = pre_saved_dict[p_id]["all_optimal_K_list"]

                # get precision
                p_r_aligned_only_correct_y_axis,wedding_to_segment_and_eventid_dict, perceived_wedding_to_segment_to_event_to_sliding_windows, recall_wedding_to_event_to_sliding_windows, recall_wed_to_event_to_best_matches = align_perceived_and_recall_event_embeds(
                                                perceived_wedding_to_segment_to_event_topic_vectors, 
                                                perceived_wedding_to_segment_to_sliding_windows, 
                                                wedding_to_order, recall_wedding_to_event_topic_vectors,
                                                                recall_wedding_to_sliding_windows,
                                                                best_matches_threshold = 0.5,
                                                                only_correct_perceived = True,
                                                                wedding_to_best_model = wedding_to_best_model)
                if p_id == 3:
                    sns.heatmap(pd.DataFrame(p_r_aligned_only_correct_y_axis[34]), vmin=0, vmax=1).set_title("Participant 3, Recall 34")
                
                overall_precision, wedding_to_precision_dict, ritual_to_avg_precision_dict, wedding_to_ritual_to_avg_precision = get_participants_row_wise_precision(p_r_aligned_only_correct_y_axis, wedding_to_segment_and_eventid_dict, perceived_wedding_to_segment_to_event_to_sliding_windows, recall_wedding_to_event_to_sliding_windows, recall_wedding_to_event_topic_vectors, perceived_wedding_to_segment_to_event_topic_vectors)
                print("got precision")
                # get within-ritual distinctiveness 
                p_r_aligned_all_perceived_y_axis,all_perceived_wedding_to_segment_and_eventid_list, all_perceived_wedding_to_segment_to_event_to_sliding_windows, recall_wedding_to_event_to_sliding_windows, recall_wed_to_event_to_best_matches = align_perceived_and_recall_event_embeds(
                                        perceived_wedding_to_segment_to_event_topic_vectors, 
                                        perceived_wedding_to_segment_to_sliding_windows, 
                                        wedding_to_order, recall_wedding_to_event_topic_vectors,
                                                        recall_wedding_to_sliding_windows,
                                                        best_matches_threshold = 0.5,
                                                        only_correct_perceived = False)
                new_saved_dict[p_id]["p_r_aligned_all_perceived_y_axis"] = p_r_aligned_all_perceived_y_axis
                new_saved_dict[p_id]["all_perceived_wedding_to_segment_and_eventid_list"] = all_perceived_wedding_to_segment_and_eventid_list
                new_saved_dict[p_id]["all_perceived_wedding_to_segment_to_event_to_sliding_windows"] = all_perceived_wedding_to_segment_to_event_to_sliding_windows
                new_saved_dict[p_id]["recall_wedding_to_event_to_sliding_windows"] = recall_wedding_to_event_to_sliding_windows
                overall_dvns, wedding_to_avg_dvns, ritual_to_avg_dvns, recall_wedding_id_to_ritual_id_to_dvns = get_participants_in_ritual_distinctiveness(p_r_aligned_all_perceived_y_axis,all_perceived_wedding_to_segment_and_eventid_list, recall_wedding_to_event_to_sliding_windows, all_perceived_wedding_to_segment_to_event_to_sliding_windows)
                print("got dvns")
                new_dict_pid = {"overall_precision": overall_precision,
                                        "wedding_to_avg_precision":  wedding_to_precision_dict,
                                        "ritual_to_avg_precision": ritual_to_avg_precision_dict,
                                        "wedding_to_ritual_to_avg_precision": wedding_to_ritual_to_avg_precision,
                                        "overall_dvns": overall_dvns,
                                        "wedding_to_avg_dvns": wedding_to_avg_dvns,
                                        "ritual_to_avg_dvns":ritual_to_avg_dvns,
                                        "recall_wedding_id_to_ritual_id_to_dvns":recall_wedding_id_to_ritual_id_to_dvns}
                participant_to_metrics[p_id] = new_dict_pid
                # json_file_path = "../../data/unfinished_business/participant_to_metrics/metrics_pid_" + str(p_id) + ".json"
                # dump_dict_to_json(json_file_path, new_dict_pid)
                participant_to_other_data[p_id] = {
                                        ""
                }
    return participant_to_metrics, new_saved_dict, all_participants_optimal_K_list




# OPTIMIZATION #

# REQUIRES: everything that trainLDA requires and other functions included here
# EFFECTS: returns a dictionary of the grid search values and the correlation values 
# NOTE: perceived_window of 5 is way too small, maybe try between [5,10,15,20,25,30]
# NOTE: window sizes above 20 for perceived can lead to each window having the same contents 
def grid_search_by_wedding_and_participant(event_seg_type = "gsbs", step_var_scale = 4, penalty_parameter = 0, stimuli_input_dir = "../../stimuli/", data_input_dir = "../../data/", num_topics_list = [50], perceived_window_length_list = [3, 5,11,21], 
                recall_window_length_list = [5], p_ids = all_pids_list, recall_k_range = range(2,50) , perceived_k_range = range(2,30)):
    correlations_list = [] # list of tuples (num_topics, perceived window size,  recall window size, performance correlation)
    # get the manual participant performance measures
    manual_performance_df =  pd.read_csv(data_input_dir + "manual_performance/wedding_recall_manual_performance.csv")
    #pdb.set_trace
    # do the grid search
    for num_topics in num_topics_list:
        for perceived_window_length in perceived_window_length_list:
            for recall_window_length in recall_window_length_list:
                print("num_topics: ", num_topics, "perceived_window_length: ", perceived_window_length, "recall_window_length: ", recall_window_length)
                # get the lda model and vectorizer for this grid
                
                lda, vectorizer, word_count_matrix, perceived_wedding_to_segment_to_sliding_windows, perceived_wedding_to_segment_id_to_embedding_dict, list_of_bags, list_of_segment_id, narrative_id = train_and_getLDA_topic_model(
                                num_components = num_topics, 
                                perceived_window_size = perceived_window_length, 
                                recall_window_size = recall_window_length)
                print("got the lda model!")
                # run the HMM on the perceived topic vectors to get event embeddings and segmentation
                
                perceived_wedding_to_segment_to_event_topic_vectors, all_perceived_optimal_K_list,_ = get_perceived_event_topic_vectors(event_seg_type, perceived_wedding_to_segment_id_to_embedding_dict, step_var_scale= step_var_scale, penalty_parameter= penalty_parameter,  k_range = perceived_k_range)
                print("completed the perceived wedding segmentation")
                # get metrics for each participant
                participant_to_metrics, new_saved_dict, all_recall_participants_optimal_K_list = get_all_participants_metrics(event_seg_type, lda_model = lda, 
                                    vectorizer = vectorizer, 
                                    perceived_wedding_to_segment_to_sliding_windows = perceived_wedding_to_segment_to_sliding_windows,
                                    perceived_wedding_to_segment_to_embeddings = perceived_wedding_to_segment_id_to_embedding_dict,
                                    perceived_wedding_to_segment_to_event_topic_vectors = perceived_wedding_to_segment_to_event_topic_vectors,
                                    p_ids = p_ids, step_var_scale = step_var_scale, penalty_parameter = penalty_parameter,
                                     recall_window_size = recall_window_length, k_range = recall_k_range)
                print("completed the recall wedding segmentation and metrics")

                # now get the correlation between automatic and manual
                participant_id_list = []
                wedding_id_list = []
                p_and_w_id_list = []
                automatic_measures_list = []
                manual_measures_list = []
                for participant_id in participant_to_metrics:
                    for couple_int in couple_int_to_wedding_id:

                        wedding_id = couple_int_to_wedding_id[couple_int]
                        # get the automatic metric
                        auto_metric = participant_to_metrics[participant_id]["wedding_to_avg_precision"][wedding_id]
                        automatic_measures_list.append(auto_metric)
                        # get the manual metric for this participant
                        query_colname = "s" + str(participant_id)
                        manual_metric = manual_performance_df[query_colname][couple_int]
                        manual_measures_list.append(manual_metric) 
                        participant_id_list.append(participant_id)
                        wedding_id_list.append(wedding_id)
                        p_and_w_id_list.append(str(participant_id) + "_" + str(wedding_id) )
                this_grid_corr = spearmanr(automatic_measures_list, manual_measures_list)[0]
                grid_parameters = {"num_topics": num_topics,
                                    "perceived_window_length": perceived_window_length,
                                    "recall_window_length": recall_window_length}
                other_data = {"automatic_measures_list":automatic_measures_list,
                              "participant_to_metrics": participant_to_metrics,
                             "manual_measures_list": manual_measures_list,
                             "participant_id_list": participant_id_list,
                             "wedding_id_list": wedding_id_list,
                             "p_and_w_id_list": p_and_w_id_list,
                             "new_saved_dict": new_saved_dict,
                             "all_perceived_optimal_K_list": all_perceived_optimal_K_list,
                             "all_recall_participants_optimal_K_list": all_recall_participants_optimal_K_list}
                correlations_list.append((this_grid_corr, grid_parameters, other_data))
    # sort this correlations list by the highest correlation
    sorted_correlations_list = sorted(correlations_list, key = lambda x: x[0])
    return sorted_correlations_list

# def grid_search_by_participant(event_seg_type, step_var_scale, penalty_parameter, k_range,stimuli_input_dir = "../../stimuli/", data_input_dir = "../../data/", num_topics_list = [5,10,25,50,75,100,125,200], perceived_window_length_list = [4,8,12,16,20,24,28,32], 
#                 recall_window_length_list = [3,5,7,9,11,13,15], p_ids = [3,4,5,30]):
#     # DEBUG: use these three modifications only for testing
#     num_topics_list = [50]
#     perceived_window_length_list = [10]
#     recall_window_length_list = [5]

#     correlations_list = [] # list of tuples (num_topics, perceived window size,  recall window size, performance correlation)
#     # get the manual participant performance measures
#     manual_performance_df =  pd.read_csv(data_input_dir + "manual_performance/wedding_recall_manual_performance.csv")
#     # do the grid search
#     for num_topics in num_topics_list:
#         for perceived_window_length in perceived_window_length_list:
#             for recall_window_length in recall_window_length_list:
#                 # get the lda model and vectorizer for this grid
#                 lda, vectorizer, word_count_matrix, perceived_wedding_to_segment_to_sliding_windows, perceived_wedding_to_segment_id_to_embedding_dict, list_of_bags, list_of_segment_id, narrative_id = train_and_getLDA_topic_model(
#                                 num_components = num_topics, 
#                                 perceived_window_size = perceived_window_length, 
#                                 recall_window_size = recall_window_length)
#                 # run the HMM on the perceived topic vectors to get event embeddings and segmentation
#                 perceived_wedding_to_segment_to_event_topic_vectors, all_optimal_K_list = get_perceived_event_topic_vectors(perceived_wedding_to_segment_id_to_embedding_dict, step_var_scale= step_var_scale, penalty_parameter= penalty_parameter,  k_range = k_range)
#                 # get metrics for each participant
#                 participant_to_metrics = get_all_participants_metrics(lda_model = lda, 
#                                     vectorizer = vectorizer, 
#                                     perceived_wedding_to_segment_to_sliding_windows = perceived_wedding_to_segment_to_sliding_windows,
#                                     perceived_wedding_to_segment_to_embeddings = perceived_wedding_to_segment_id_to_embedding_dict,
#                                     perceived_wedding_to_segment_to_event_topic_vectors = perceived_wedding_to_segment_to_event_topic_vectors,
#                                     p_ids = p_ids, step_var_scale = step_var_scale, penalty_parameter = penalty_parameter,
#                                      recall_window_size = recall_window_length)
#                 # now get the correlation between automatic and manual
#                 participant_id_list = []
#                 automatic_measures_list = []
#                 manual_measures_list = []
#                 for participant_id in participant_to_metrics:
#                     # get the automatic metric
#                     auto_metric = participant_to_metrics[participant_id]["overall_precision"]
#                     automatic_measures_list.append(auto_metric)
#                     # get the manual metric for this participant
#                     query_colname = "s_" + str(participant_id)
#                     manual_metric = manual_performance_df[query_colname][0]
#                     manual_measures_list.append(manual_metric) 
#                 this_grid_corr = pearsonr(automatic_measures_list, manual_measures_list)[0]
#                 correlations_list.append((num_topics, perceived_window_length, recall_window_length, this_grid_corr))
#     # sort this correlations list by the highest correlation
#     sorted_correlations_list = sorted(correlations_list, key = lambda x: x[3])
#     return sorted_correlations_list
# PLOTS #

# get figure 2b or 2e in Manning paper without overlay of the episode
def plot_windows_by_embedding_self_correlation_matrix(matrix,side_label = "Window", title = "Window-Window Topic Proportions Correlation"):
    f, ax = plt.subplots(1,1, figsize = (10,8))
    img = ax.imshow(matrix, cmap='viridis')
    f.colorbar(img)
    ax.set_title(title)
    ax.set_xlabel(side_label)
    ax.set_ylabel(side_label)

# REQUIRES: data_matrix which contains the topic vectors of interest
# bounds which contains the indices where one event switches to another
# n_TRs is the number of rows
def plot_tt_similarity_matrix(data_matrix, bounds, n_TRs, title_text):
    f, ax = plt.subplots(1,1, figsize = (7,5))
    img = ax.imshow(np.corrcoef(data_matrix), cmap='viridis', vmin=0, vmax=1)
    ax.set_title(title_text)
    ax.set_xlabel('sliding window embedding')
    ax.set_ylabel('sliding window embedding')
    f.colorbar(img)
    # plot the boundaries 
    # make the non-zero indices be increased by 1 so that they start at the next
    # event 
    new_bounds_list = []
    for bo in bounds:
        new_bo = bo + 1 if bo != -1 else 0
        new_bounds_list.append(new_bo)
    bounds = new_bounds_list 
    bounds_aug = bounds
    for i in range(len(bounds_aug)-1):
        rect = patches.Rectangle(
            (bounds_aug[i],bounds_aug[i]),
            bounds_aug[i+1]-bounds_aug[i],
            bounds_aug[i+1]-bounds_aug[i],
            linewidth=2,edgecolor='w',facecolor='none'
        )
        ax.add_patch(rect)

# EFFECTS: outputs the top words of each topic
def plot_model_topics(lda_model, vectorizer, n_top_words, title):
    feature_names = vectorizer.get_feature_names()
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(lda_model.components_[0:10]):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def plot_precision_on_dnvs_by_wedding_and_ritual(participant_to_metrics):
    overall_dvns_list = [] 
    overall_prec_list = []
    pid_list = []
    for pid in participant_to_metrics:
        for wedding_id in participant_to_metrics[pid]['wedding_to_ritual_to_avg_precision']:
            for ritual_id in participant_to_metrics[pid]['wedding_to_ritual_to_avg_precision'][wedding_id]:
                new_prec = participant_to_metrics[pid]['wedding_to_ritual_to_avg_precision'][wedding_id][ritual_id]
                new_dvns = participant_to_metrics[pid]['recall_wedding_id_to_ritual_id_to_dvns'][wedding_id][ritual_id]
                overall_dvns_list.append(new_dvns)
                overall_prec_list.append(new_prec)
                pid_list.append(pid)
    print(pearsonr(overall_dvns_list,overall_prec_list))
    d = {"overall_dvns": overall_dvns_list, "overall_precision": overall_prec_list, "pids": pid_list}
    df = pd.DataFrame(data = d)
    (ggplot(df) 
         + geom_point(aes(x="overall_dvns", y="overall_precision"))
      # + geom_label(aes(x="overall_dvns", y="overall_precision",label="factor(pids)"))
    ).draw()
    