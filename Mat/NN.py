#svd where test set = validation set

import json
from pathlib import Path
from sklearn.svm import SVR
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score, accuracy_score




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



# #small 2
# training_set = ['ES2002', 'ES2005', 'ES2006', 'IS1000', 'IS1001', 'IS1002', 'TS3005', 'TS3008']
# training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
# training_set.remove('IS1002a')






# #small 2bis
# training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003' ,'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010']
# training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
# training_set.remove('IS1002a')


# #small 3
# training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011']
# training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
# training_set.remove('IS1002a')
# training_set.remove('IS1005d')


##all
training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')






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

    
    X_training_previous.append("")
    for j in range(1, len(X_training_text_current)):
        X_training_previous.append(X_training_text_current[X_training_previous_number[j]])

    y_training += training_labels[transcription_id]

print("start encoding training ") 
X_training = bert.encode(X_training, show_progress_bar=True)
X_training_edges = bert.encode(X_training_edges, show_progress_bar=True)
X_training_speaker = bert.encode(X_training_speaker, show_progress_bar=True)
X_training_previous = bert.encode(X_training_previous, show_progress_bar=True)
print("end encoding")


output_folder = 'embeddings_training_All/'

# Assurez-vous que le dossier d'output existe
import os
os.makedirs(output_folder, exist_ok=True)

# Définissez les noms de fichiers de sortie
file_names = ['X_training.json', 'X_training_edges.json', 'X_training_speaker.json', 'X_training_previous.json']

# Les embeddings que vous souhaitez enregistrer
embeddings_to_save = [X_training, X_training_edges, X_training_speaker, X_training_previous]

# Enregistrez les embeddings dans des fichiers JSON
for file_name, embeddings in zip(file_names, embeddings_to_save):
    output_path = os.path.join(output_folder, file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings.tolist(), f)

# Affichez un message une fois que c'est fait
print("Embeddings enregistrés avec succès dans le dossier embeddings.")












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




# text_SVD: utterances are embedded with SentenceTransformer, then train a support vector regression model.
bert = SentenceTransformer('all-MiniLM-L6-v2')



y_test = []
with open("training_labels.json", "r") as file:
    training_labels = json.load(file)
X_test = [] 
X_test_edges = []
X_test_speaker = []
X_test_previous = []

for transcription_id in test_set:
    X_test_edges.append("Beginning")
    
    file_path = "training" + "/" + transcription_id + '.txt'
    # Open the file in read mode


    X_test_previous_number = []
    X_test_previous_number.append(0)


    


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

            
            X_test_previous_number.append(int(words[0]))

            c_line += 1
    

    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)


    X_test_text_current = []



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
        
        X_test_speaker.append(speaker)


        if(c == 0 ):
            X_test.append(utterance["text"])
            X_test_text_current.append(utterance["text"])
        else:
            X_test.append(utterance["text"])
            X_test_edges.append(edge_type_transcriptionId[c-1])
            X_test_text_current.append(utterance["text"])
        c+=1


    #print("X test text current" , X_test_text_current)
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




output_folder = 'embeddings_test_Small2bis/'

# Assurez-vous que le dossier d'output existe
import os
os.makedirs(output_folder, exist_ok=True)

# Définissez les noms de fichiers de sortie
file_names = ['X_test.json', 'X_test_edges.json', 'X_test_speaker.json', 'X_test_previous.json']

# Les embeddings que vous souhaitez enregistrer
embeddings_to_save = [X_test, X_test_edges, X_test_speaker, X_test_previous]

# Enregistrez les embeddings dans des fichiers JSON
for file_name, embeddings in zip(file_names, embeddings_to_save):
    output_path = os.path.join(output_folder, file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings.tolist(), f)

# Affichez un message une fois que c'est fait
print("Embeddings enregistrés avec succès dans le dossier embeddings.")










########################################################## Model #######################################################################



# Supposons que vous avez déjà vos embeddings pour chaque catégorie
embedding_dim = X_training_edges.shape[1]
num_categories = 4  # Nombre de catégories d'embedding

# Entrées
input_text_embedding = layers.Input(shape=(embedding_dim,), name='text_embedding')
input_title_embedding = layers.Input(shape=(embedding_dim,), name='title_embedding')
input_link_embedding = layers.Input(shape=(embedding_dim,), name='link_embedding')
input_prev_sentence_embedding = layers.Input(shape=(embedding_dim,), name='prev_sentence_embedding')

# Concaténation des embeddings
concatenated = layers.concatenate([input_text_embedding, input_title_embedding, input_link_embedding, input_prev_sentence_embedding])

# Ajout de couches denses
dense1 = layers.Dense(256, activation='relu')(concatenated)
dense2 = layers.Dense(128, activation='relu')(dense1)

# Couche de sortie
output = layers.Dense(1, activation='sigmoid', name='output')(dense2)

# Création du modèle
model = models.Model(inputs=[input_text_embedding, input_title_embedding, input_link_embedding, input_prev_sentence_embedding], outputs=output)

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Afficher le résumé du modèle
model.summary()














########################################################## Fit #######################################################################




X_train_text = X_training 
X_train_title = X_training_edges 
X_train_link = X_training_speaker 
X_train_prev_sentence = X_training_previous 
y_train = y_training




X_val_text = X_test 
X_val_title = X_test_edges 
X_val_link = X_test_speaker 
X_val_prev_sentence = X_test_previous 
y_val = y_test





# Convert your data to numpy arrays
X_train_text = np.array(X_train_text)
X_train_title = np.array(X_train_title)
X_train_link = np.array(X_train_link)
X_train_prev_sentence = np.array(X_train_prev_sentence)
y_train = np.array(y_train)

X_val_text = np.array(X_val_text)
X_val_title = np.array(X_val_title)
X_val_link = np.array(X_val_link)
X_val_prev_sentence = np.array(X_val_prev_sentence)
y_val = np.array(y_val)

# ... (the rest of your previous code)

# Entraînement du modèle
history = model.fit(
    [X_train_text, X_train_title, X_train_link, X_train_prev_sentence],
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=([X_val_text, X_val_title, X_val_link, X_val_prev_sentence], y_val)
)



model.save_weights('NN_small3.h5')



print("starting predictions")
predictions = model.predict([X_val_text, X_val_title, X_val_link, X_val_prev_sentence])
binary_predictions = (predictions > 0.5).astype(int)



for i in range(len(binary_predictions)):
    print(f"Exemple {i + 1} - Prédiction : {binary_predictions[i]}, Vraie étiquette : {y_test[i]}")



# Calcul du F1 score
f1 = f1_score(binary_predictions, y_test)

# Affichage du F1 score
print(f'F1 Score on Training Data: {f1}')
