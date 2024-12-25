import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = pd.read_csv('data/HateTunisianData.csv')

X = data['hatespeech']
y = data['index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

best_model = joblib.load('src/hate_speech_model.pkl')

y_pred = best_model.predict(vectorizer.transform(X_test))

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report_df.to_csv('performance_metrics/performance_metrics.csv')
print(report_df) 