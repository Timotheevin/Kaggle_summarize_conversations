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
path_to_test = Path("test")  #be careful it's changed to test here



#####
# training and test sets of transcription ids
#####
training_set = ['ES2002', 'ES2005', 'ES2006', 'IS1000', 'IS1001', 'IS1002', 'TS3005', 'TS3008']



training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
# training_set.remove('IS1005d')
# training_set.remove('TS3012c')

test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])




# text_SVD: utterances are embedded with SentenceTransformer, then train a support vector regression model.
bert = SentenceTransformer('all-MiniLM-L6-v2')


svr = joblib.load('svr_model_small3_vGraph_case.joblib')
print("weights loaded")


threshold = 0.28

test_labels = {}
for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)

    file_path = "test" + "/" + transcription_id + '.txt'
    # Open the file in read mode
    with open(file_path, 'r') as file:
        edge_type_transcriptionId = []
        # Read each line of the file
        for line in file:
            # Split the line into words using space as a separator
            words = line.split()

            # If the line is not empty and contains at least three words
            if words:
                edge_type_transcriptionId.append(words[1])    # Type of the edge

    X_test = []
    X_test_edges = ["Beginning"]
    print(transcription_id)
    c = 0
    for utterance in transcription:
        if(c == 0 ):
            X_test.append(utterance["speaker"] + ": " + utterance["text"])
        else:
            X_test.append(utterance["speaker"] + ": " + utterance["text"])
            X_test_edges.append(edge_type_transcriptionId[c-1])
        c+=1

    X_test = bert.encode(X_test)
    X_test_edges = bert.encode(X_test_edges, show_progress_bar=True)
    print("encoding training dataset finished")

    #print("Encoding test set finished on " + transcription_id)
    X_test_final = [np.concatenate((x, y)) for x, y in zip(X_test, X_test_edges)]


    y_test = svr.predict(X_test_final)
    #print("prediction on test set finished")

    # You may need to convert the continuous predictions to binary labels if needed
    # For example, if y_test > threshold, label as 1, else label as 0
    binary_labels = [1 if pred > threshold else 0 for pred in y_test]

    test_labels[transcription_id] = binary_labels



with open("test_labels_text_svr_small3_vGraph_case.json", "w") as file:
    json.dump(test_labels, file, indent=4)



