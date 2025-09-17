import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier

df = pd.read_csv('breast_cancer.csv')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training CatBoost Model
classifier = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    random_state=0,
    verbose=0
)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracies = cross_val_score(classifier, X_train, y_train, cv=10)
print("Accuracy Mean: ", accuracies.mean())

print("Accuracy Standard Deviation : ", accuracies.std())
