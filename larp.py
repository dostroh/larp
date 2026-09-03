import os
from pyexpat import features
from random import Random

import kagglehub
import numpy
import pandas
import sklearn
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

path=kagglehub.dataset_download("yasserh/titanic-dataset")
csv_path=os.path.join(path , "Titanic-Dataset.csv")
df = pandas.read_csv(csv_path)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
x = pandas.get_dummies(
    df[features].fillna(df[["Age", "Fare"]].median()), drop_first=True
)
y=df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

model=RandomForestClassifier(1000, random_state=7)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")