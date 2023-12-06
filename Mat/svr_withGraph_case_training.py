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

#####
# training and test sets of transcription ids
#####



# #small 1 (only to test code)
# training_set = ['ES2002']
# training_set = flatten([[m_id+s_id for s_id in 'ab'] for m_id in training_set])



#small 2
training_set = ['ES2002', 'ES2005', 'ES2006', 'IS1000', 'IS1001', 'IS1002', 'TS3005', 'TS3008']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')



# #small 3
# training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011']
# training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
# training_set.remove('IS1002a')
# #training_set.remove('IS1005d')
# #training_set.remove('TS3012c')






# text_SVD: utterances are embedded with SentenceTransformer, then train a support vector regression model.
bert = SentenceTransformer('all-MiniLM-L6-v2')



y_training = []
with open("training_labels.json", "r") as file:
    training_labels = json.load(file)
X_training = [] 
X_training_edges = []
X_training_speaker = []
X_training_previous = []

for transcription_id in training_set:
    X_training_edges.append("Beginning")
    
    file_path = "training" + "/" + transcription_id + '.txt'
    # Open the file in read mode


    X_training_previous_number = []
    X_training_previous_number.append(0)


    


    with open(file_path, 'r') as file:
        edge_type_transcriptionId = []
        # Read each line of the file
        c_line = 0
        for line in file:
            # Split the line into words using space as a separator
            words = line.split()

            # If the line is not empty and contains at least three words
            if words:
                edge_type_transcriptionId.append(words[1])    # Type of the edge
            else:
                print("WORDSSSSSSSSSSSSSSSSSSSSSSS")

            
            X_training_previous_number.append(int(words[0]))

            c_line += 1
    

    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)


    X_training_text_current = []



    c = 0
    for utterance in transcription:
        if(utterance["speaker"]  == "PM"):
            speaker = "Project Manager"
        elif(utterance["speaker"]  == "ME"):
            speaker = "Marketing Expert"
        elif(utterance["speaker"]  == "UI"):
            speaker = "Interface Designer"
        else:
            speaker = "Industrial Desginer"
        
        X_training_speaker.append(speaker)


        if(c == 0 ):
            X_training.append(utterance["text"])
            X_training_text_current.append(utterance["text"])
        else:
            X_training.append(utterance["text"])
            X_training_edges.append(edge_type_transcriptionId[c-1])
            X_training_text_current.append(utterance["text"])
        c+=1


    #print("X training text current" , X_training_text_current)
    print(len(X_training_text_current))
    X_training_previous.append("")
    for j in range(1, len(X_training_text_current)):
        X_training_previous.append(X_training_text_current[X_training_previous_number[j]])

    y_training += training_labels[transcription_id]

print("start encoding") 
X_training = bert.encode(X_training, show_progress_bar=True)


X_training_edges = bert.encode(X_training_edges, show_progress_bar=True)
X_training_speaker = bert.encode(X_training_speaker, show_progress_bar=True)
X_training_previous = bert.encode(X_training_previous, show_progress_bar=True)

X_training_final = [np.concatenate((x, y, z ,a)) for x, y,z,a in zip(X_training_speaker, X_training, X_training_edges, X_training_previous)]
print("final premier element ", X_training_final[0])
print("encoding training dataset finished")
# Use Support Vector Regression


print(len(X_training))
print(len(X_training_edges))
print(len(X_training_final))
print(len(y_training))

svr = SVR(verbose=True)
svr.fit(X_training_final, y_training)

# with tqdm1(total=len(X_training), desc="Training SVR", unit="batch") as progress:
#     # Utilisez la fonction partial pour obtenir une fonction qui met à jour la barre de progression
#     partial_update_progress = tqdm1.write if progress.dynamic_ncols else progress.update
#     svr = SVR(verbose=True)
#     svr.fit(X_training, y_training, callback=lambda x: partial_update_progress(x['n_samples']))

print("fit (on training dataset) finished")

joblib.dump(svr, 'svr_model_small2_vGraph_case_newWay.joblib')
print("weights saved")




