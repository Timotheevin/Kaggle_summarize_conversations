#save embeddings of the training set and the test set
#output: embedding of the text sentence (X_training.json), embedding of the speaker title: (X_training_edges.json), embedding of the previous sentence: (X_training_previous.json), embedding of the previous sentence: (X_training_previous.json)


import json
from pathlib import Path
from sklearn.svm import SVR
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score, accuracy_score
import os




########################################################## Params #######################################################################

output_folder_training = 'embeddings_training_Small2bis/'
output_folder_test = 'embeddings_test_Small2bis/'



def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

path_to_training = Path("training")
path_to_test = Path("training")  #be careful it's changed to training here






########################################################## Training Data #######################################################################

# #small 1 (only to test code)
# training_set = ['ES2002']
# training_set = flatten([[m_id+s_id for s_id in 'ab'] for m_id in training_set])


# #small 2
# training_set = ['ES2002', 'ES2005', 'ES2006', 'IS1000', 'IS1001', 'IS1002', 'TS3005', 'TS3008']
# training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
# training_set.remove('IS1002a')


#small 2bis
training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003' ,'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')


# #small 3
# training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011']
# training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
# training_set.remove('IS1002a')
# training_set.remove('IS1005d')


# ##all
# training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
# training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
# training_set.remove('IS1002a')
# training_set.remove('IS1005d')
# training_set.remove('TS3012c')






#utterances are embedded with SentenceTransformer
bert = SentenceTransformer('all-MiniLM-L6-v2')



with open("training_labels.json", "r") as file:
    training_labels = json.load(file)

y_training = []
X_training = [] 
X_training_edges = []
X_training_speaker = []
X_training_previous = []
X_training_length = []

for transcription_id in training_set:
    X_training_previous_number = []  #to keep track of the index of the previous node in the graph
    X_training_previous_number.append(0)
    X_training_edges.append("Beginning")

    
    file_path = Path("training") / f"{transcription_id}.txt"


    
    #####Process the edges data
    with open(file_path, 'r') as file:
        edge_type_transcriptionId = []

    for c_line, line in file:
        words = line.split()
        if words:
            edge_type = words[1]
            edge_type_transcriptionId.append(edge_type)

            source_node = words[0]
            target_node = words[2]

        X_training_previous_number.append(int(source_node))





    #####Process the nodes data
    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)


    X_training_text_current = []


    for c, utterance in enumerate(transcription):

        #speaker
        speaker_mapping = {
            "PM": "Project Manager",
            "ME": "Marketing Expert",
            "UI": "Interface Designer",
            "ID": "Industrial Designer"
        }
        speaker = speaker_mapping.get(utterance["speaker"])
        X_training_speaker.append(speaker)


        #length
        X_training_length.append(len(utterance["text"]))


        #text
        X_training.append(utterance["text"])
        X_training_text_current.append(utterance["text"])


        #edges
        if c > 0:
            X_training_edges.append(edge_type_transcriptionId[c - 1])
        

    #previous sentence
    X_training_previous.append("")
    for j in range(1, len(X_training_text_current)):
        X_training_previous.append(X_training_text_current[X_training_previous_number[j]])


    #labels
    y_training += training_labels[transcription_id]



#Encoding
print("start encoding training ") 
X_training = bert.encode(X_training, show_progress_bar=True)
X_training_edges = bert.encode(X_training_edges, show_progress_bar=True)
X_training_speaker = bert.encode(X_training_speaker, show_progress_bar=True)
X_training_previous = bert.encode(X_training_previous, show_progress_bar=True)
print("end encoding")






os.makedirs(output_folder_training, exist_ok=True)

file_names = ['X_training.json', 'X_training_edges.json', 'X_training_speaker.json', 'X_training_previous.json','X_training_length.json']

embeddings_to_save = [X_training, X_training_edges, X_training_speaker, X_training_previous, X_training_length]

for file_name, embeddings in zip(file_names, embeddings_to_save):
    output_path = os.path.join(output_folder_training, file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)


print("Embeddings successfully saved.")












########################################################## Validation Data #######################################################################


path_to_training = Path("training")
path_to_test = Path("training")  #be careful it's changed to training here

#####
# training and test sets of transcription ids
#####



# #small 1 (only to test code)
# test_set = ['ES2013']
# test_set = flatten([[m_id+s_id for s_id in 'ab'] for m_id in test_set])



# #small 2
# test_set = ['ES2013', 'IS1004', 'TS3012']
# test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])
# test_set.remove('TS3012c')


#small 2 bis
test_set = ['ES2012', 'ES2013', 'IS1004','IS1005', 'TS3011', 'TS3012']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])
test_set.remove('TS3012c')
test_set.remove('IS1005d')




bert = SentenceTransformer('all-MiniLM-L6-v2')

y_test = []
X_test = []
X_test_edges = []
X_test_speaker = []
X_test_previous = []
X_test_length = []

for transcription_id in test_set:
    X_test_edges.append("Beginning")

    file_path = "training" / f"{transcription_id}.txt"

    X_test_previous_number = [0]

    with open(file_path, 'r') as file:
        edge_type_transcriptionId = [words[1] for line in file if (words := line.split())]

        for words in file:
            if words:
                edge_type_transcriptionId.append(words[1])

            X_test_previous_number.append(int(words[0]))

    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)

    X_test_text_current = []

    for c, utterance in enumerate(transcription):
        speaker_mapping = {"PM": "Project Manager", "ME": "Marketing Expert", "UI": "Interface Designer"}
        speaker = speaker_mapping.get(utterance["speaker"], "Industrial Designer")
        X_test_speaker.append(speaker)
        X_test_length.append(len(utterance["text"]))

        X_test.append(utterance["text"])
        X_test_text_current.append(utterance["text"])

        if c > 0:
            X_test_edges.append(edge_type_transcriptionId[c - 1])

    X_test_previous.append("")
    for j in range(1, len(X_test_text_current)):
        X_test_previous.append(X_test_text_current[X_test_previous_number[j]])

    y_test += training_labels[transcription_id]


print("start encoding test ") 
X_test = bert.encode(X_test, show_progress_bar=True)
X_test_edges = bert.encode(X_test_edges, show_progress_bar=True)
X_test_speaker = bert.encode(X_test_speaker, show_progress_bar=True)
X_test_previous = bert.encode(X_test_previous, show_progress_bar=True)
print("end encoding")




###Save embeddings in JSON files
os.makedirs(output_folder_test, exist_ok=True)

file_names = ['X_test.json', 'X_test_edges.json', 'X_test_speaker.json', 'X_test_previous.json','X_test_length.json']


embeddings_to_save = [X_test, X_test_edges, X_test_speaker, X_test_previous,X_test_length]

for file_name, embeddings in zip(file_names, embeddings_to_save):
    print("-")
    output_path = os.path.join(output_folder_test, file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)


print("Embeddings successfully saved.")
