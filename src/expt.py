import dvc.api
import pandas
import numpy as np
from numpy import savetxt
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

with dvc.api.open(repo="https://github.com/Rushi-101/MLOps_Assignment", path="data/creditcard.csv", mode="r") as fd:
	df = pandas.read_csv(fd)

data = df.values

X = data[:, 0:29].astype(str)
y = data[:, 30].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21, shuffle= True)

savetxt('data/prepared/train.csv', X_train, delimiter=',', fmt='%s')
savetxt('data/prepared/test.csv', X_test, delimiter=',', fmt='%s')

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

random_forest_pkl_filename = 'models/model.pkl'
random_forest_model_pkl = open(random_forest_pkl_filename, 'wb')
pickle.dump(classifier, random_forest_model_pkl)

y_pred = classifier.predict(X_test)
accuracy_score = accuracy_score(y_test,y_pred)
f1_score = f1_score(y_test,y_pred,average='macro')

print("Accuracy: ", accuracy_score)
print("Macro F1 score: ", f1_score)

accuracy_file = open("metrics/accuracy.json", "w") 
json.dump(accuracy_score, accuracy_file, indent = 6) 
f1_score_file = open("metrics/f1_score.json", "w") 
json.dump(f1_score, f1_score_file, indent = 6) 