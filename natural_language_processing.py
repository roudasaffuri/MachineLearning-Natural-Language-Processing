"""
# Natural Language Processing

## Importing the libraries
"""


import numpy as np
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

"""## Cleaning the texts"""

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  # [^a-zA-Z]  That is NOT a Letter
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

print(corpus)

"""## Creating the Bag of Words model"""

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""## Training the SVM model on the Training set"""

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)



# Function to preprocess and predict a single sentence
def preprocess_text(text):
  text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
  return ' '.join(text)


# Function to transform a single preprocessed sentence
def transform_text(text, vectorizer):
  return vectorizer.transform([text]).toarray()


# Function to preprocess and predict a single sentence
def predict_sentence(sentence):
  processed_sentence = preprocess_text(sentence)
  transformed_sentence = transform_text(processed_sentence, cv)
  prediction = classifier.predict(transformed_sentence)
  return prediction


# Input a sentence to test
input_sentence = input("Enter a sentence to check: ")
prediction = predict_sentence(input_sentence)

# Output the prediction
print("Prediction for the entered sentence:", "Positive" if prediction == 1 else "Negative")