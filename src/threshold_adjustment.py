import pandas as pd
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np

data = pd.read_csv('data/HateTunisianData.csv')

X = data['hatespeech']
y = data['index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

best_model = joblib.load('src/hate_speech_model.pkl')

y_scores = best_model.predict_proba(vectorizer.transform(X_test))[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

desired_recall = 0.75
threshold_index = np.argmax(recall >= desired_recall)
optimal_threshold = thresholds[threshold_index]

y_pred_adjusted = (y_scores >= optimal_threshold).astype(int)

print("Adjusted Classification Report:")
print(classification_report(y_test, y_pred_adjusted))

print(confusion_matrix(y_test, y_pred_adjusted)) 