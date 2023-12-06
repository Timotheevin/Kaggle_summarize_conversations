import json
from pathlib import Path

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

def make_predictions(X_test, clf):

    test_labels = {}
    for transcription_id in test_set:
        with open(path_to_test / f"{transcription_id}.json", "r") as file:
            transcription = json.load(file)
        
        X_test = []
        for utterance in transcription:
            X_test.append(utterance["speaker"] + ": " + utterance["text"])
        
        X_test = bert.encode(X_test)

        y_test = clf.predict(X_test)
        test_labels[transcription_id] = y_test.tolist()

    with open("test_labels_text_baseline.json", "w") as file:
        json.dump(test_labels, file, indent=4)

path_to_training = Path("training")
path_to_test = Path("test")

#####
# training and test sets of transcription ids
#####

test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])

#####
# SVM
#####
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle

bert = SentenceTransformer('all-MiniLM-L6-v2')

# load variables from pickle file

X_training = pickle.load(open("../data/X_training.pkl", "rb"))
y_training = pickle.load(open("../data/y_training.pkl", "rb"))

print(X_training.shape)

# split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.2, random_state=42)

print("split done")

# create a svm classifier

clf = SVC(kernel='rbf', verbose=True, class_weight='balanced')

# Train the model using the training sets

clf.fit(X_train, y_train)

print("fit done")

# Predict the response for test dataset

y_pred = clf.predict(X_test)

print("predict done")

# compute F1 score

print(f1_score(y_test, y_pred))


