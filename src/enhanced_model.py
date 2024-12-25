import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

data = pd.read_csv('data/HateTunisianData.csv')

X = data['hatespeech']
y = data['index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]  # Smoothing parameter for Naive Bayes
}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_vectorized, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_vectorized)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

joblib.dump(best_model, 'src/hate_speech_model.pkl')
joblib.dump(vectorizer, 'src/vectorizer.pkl') 