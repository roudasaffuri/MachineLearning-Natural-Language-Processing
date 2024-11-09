# Sentiment Analysis on Restaurant Reviews
This program performs sentiment analysis on restaurant reviews using Natural Language Processing (NLP) and Support Vector Machine (SVM) classification. It classifies reviews as either positive or negative and includes a feature to predict the sentiment of custom input sentences.

## Program Overview
Load Dataset: Loads a TSV file of restaurant reviews labeled as positive or negative.

Text Preprocessing: Cleans the text by removing non-letters, converting to lowercase, removing stopwords, and stemming words to their root forms.

Bag-of-Words Model: Converts reviews to a structured format with word frequencies, capturing the most common 1500 words.

Train-Test Split: Splits data to train and test the model separately.

Train SVM Model: Trains an SVM classifier on the review data to predict sentiment.

Evaluate Model: Outputs accuracy and a confusion matrix to assess model performance.

Custom Sentence Prediction: Allows the user to input a sentence and returns whether it is predicted as positive or negative.

## Example Usage
Run the program and enter a sentence when prompted, like "The service was fantastic!".
The output will be "Positive" if the review is classified as positive, or "Negative" otherwise.
