import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("breast_cancer.csv")
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_val)
print(y_pred)

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix: ", cm)
acc = cross_val_score(classifier, X_train, y_train, cv=10)
print("Accuracy: %", acc.mean()*100)
print("Standard Deviation: %", acc.std()*100)