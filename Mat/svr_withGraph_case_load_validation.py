#svd where test set = validation set

import json
from pathlib import Path
from sklearn.svm import SVR
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

path_to_training = Path("training")
path_to_test = Path("training")  #be careful it's changed to training here


# #small 1 (only to test code)
# #test_set = ['ES2008', 'IS1004', 'TS3011']
# test_set = ['ES2013']
# test_set = flatten([[m_id+s_id for s_id in 'bcd'] for m_id in test_set])


#small 2
test_set = ['ES2013', 'IS1004', 'TS3012']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])
test_set.remove('TS3012c')

# text_SVD: utterances are embedded with SentenceTransformer, then train a support vector regression model.
bert = SentenceTransformer('all-MiniLM-L6-v2')


svr = joblib.load('svr_model_small2_vGraph_case_newWay.joblib')
print("weights loaded")



with open('training_labels.json', 'r', encoding='utf-8') as file:
    real_labels_data = json.load(file)

threshold = 0.35

result_list_global = {}
test_labels = {}

true_positive_global = 0
trueAndFalse_positive_global = 0
real_positive_global = 0


test_labels = {}
for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)

    file_path = "training" + "/" + transcription_id + '.txt'
    # Open the file in read mode

    X_test_previous_number = []
    X_test_previous_number.append(0)

    with open(file_path, 'r') as file:
        edge_type_transcriptionId = []
        # Read each line of the file
        for line in file:
            # Split the line into words using space as a separator
            words = line.split()

            # If the line is not empty and contains at least three words
            if words:
                edge_type_transcriptionId.append(words[1])    # Type of the edge

            X_test_previous_number.append(int(words[0]))

    X_test = []
    X_test_edges = ["Beginning"]
    X_test_speaker = []
    X_test_previous = []


    X_test_text_current = []


    print(transcription_id)
    c = 0
    for utterance in transcription:
        if(c == 0 ):
            X_test.append(utterance["text"])
        else:
            X_test.append(utterance["text"])
            X_test_edges.append(edge_type_transcriptionId[c-1])
        c+=1

        if(utterance["speaker"]  == "PM"):
            speaker = "Project Manager"
        elif(utterance["speaker"]  == "ME"):
            speaker = "Marketing Expert"
        elif(utterance["speaker"]  == "UI"):
            speaker = "Interface Designer"
        else:
            speaker = "Industrial Desginer"
            
        X_test_speaker.append(speaker)

        X_test_previous.append("")
        for j in range(1, len(X_test_text_current)):
            X_test_previous.append(X_test_text_current[X_test_previous_number[j]])
        
    
    X_test = bert.encode(X_test, show_progress_bar=True)
    X_test_edges = bert.encode(X_test_edges, show_progress_bar=True)
    X_test_speaker = bert.encode(X_test_speaker, show_progress_bar=True)
    X_test_previous = bert.encode(X_test_previous, show_progress_bar=True)

    # print("encoding training dataset finished")

    # #print("Encoding test set finished on " + transcription_id)

    # #print shape of the four lists with a text indacating their names
    # print("X_test shape: ", X_test.shape)
    # print("X_test_edges shape: ", X_test_edges.shape)
    # print("X_test_speaker shape: ", X_test_speaker.shape)
    # print("X_test_previous shape: ", X_test_previous.shape)


    

    X_test_final = [np.concatenate((x, y, z ,a)) for x, y,z,a in zip(X_test_speaker, X_test, X_test_edges, X_test_previous)]


    y_test = svr.predict(X_test_final)
    #print("prediction on test set finished")

    # You may need to convert the continuous predictions to binary labels if needed
    # For example, if y_test > threshold, label as 1, else label as 0
    binary_labels = [1 if pred > threshold else 0 for pred in y_test]

    test_labels[transcription_id] = binary_labels

    real_labels_for_one_transcriptionID = real_labels_data[transcription_id]

    intersection = np.logical_and(binary_labels, real_labels_for_one_transcriptionID)

    count_True_Positive = np.sum(intersection)

    precision = count_True_Positive / np.sum(binary_labels)
    recall = count_True_Positive  / np.sum(real_labels_for_one_transcriptionID)
    F1 = 2 * precision * recall / (precision + recall)

    true_positive_global += count_True_Positive
    trueAndFalse_positive_global += np.sum(binary_labels)
    real_positive_global += np.sum(real_labels_for_one_transcriptionID)
    print("precision: ", precision)
    print("recall: ",  recall)
    print(F1)


print()
print()
print("Global result for threshold: ", threshold)

precision = true_positive_global / trueAndFalse_positive_global
recall = true_positive_global  / real_positive_global
F1 = 2 * precision * recall / (precision + recall)

print("precision: ", precision)
print("recall: ",  recall)
print("F1: ", F1)

with open("test_labels_text_svr_small2_vGraph_case_new24.json", "w") as file:
    json.dump(test_labels, file, indent=4)

