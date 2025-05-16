# 2.1 Import Libraries
# ...
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
# 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2.2 Load the data
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

data_fake["class"] = 0
data_true["class"] = 1

data_fake_manual_testing = data_fake.tail(10)
data_true_manual_testing = data_true.tail(10)

data_fake = data_fake.iloc[:-10]
data_true = data_true.iloc[:-10]

data_merge = pd.concat([data_fake, data_true], axis=0)
data_merge = data_merge.sample(frac=1).reset_index(drop=True)  # Shuffle the data

def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'http.?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data_merge['text'] = data_merge['title'] + " " + data_merge['text']
data_merge['text'] = data_merge['text'].apply(text_cleaning)

# 2.7 Prepare training and testing data
x = data_merge['text']
y = data_merge['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# 2.8 TF-IDF Vectorization
vectorization = TfidfVectorizer(max_features=5000)  # Added limit for better performance
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# 2.9 Logistic Regression Model
model = LogisticRegression()
model.fit(xv_train, y_train)

# 2.10 Evaluation
y_pred = model.predict(xv_test)

print("Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.show()

# 2.11 Manual Testing (on the set aside 20 news)
data_manual_testing = pd.concat([data_fake_manual_testing, data_true_manual_testing], axis=0)

data_manual_testing['text'] = data_manual_testing['title'] + " " + data_manual_testing['text']
data_manual_testing['text'] = data_manual_testing['text'].apply(text_cleaning)

manual_testing_vectors = vectorization.transform(data_manual_testing['text'])
manual_testing_predictions = model.predict(manual_testing_vectors)

# Mapping function
def label_to_text(label):
    return "Fake" if label == 0 else "True"

manual_testing_results = pd.DataFrame({
    'Title': data_manual_testing['title'],
    'Predicted Label': [label_to_text(x) for x in manual_testing_predictions],
    'Actual Label': [label_to_text(x) for x in data_manual_testing['class']]
})

print("\nManual Testing Results:")
print(manual_testing_results)
