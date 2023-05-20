from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer

import tensorflow as tf

import os

# # Get the absolute path to the current directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
#
# # Construct the path to the imdb_bert_pretrained folder
# model_path = os.path.join(current_dir, 'imdb_bert_pretrained')

# Get the path to the model directory
model_path = os.path.join(os.path.dirname(__file__), 'imdb_bert_pretrained')



# Load the tokenizer and model outside the route function to improve performance
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
best_model = TFBertForSequenceClassification.from_pretrained(model_path)



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        url = request.form['url']
        # Perform web scraping using BeautifulSoup
        webpage = requests.get(url)
        soup = BeautifulSoup(webpage.content, "html.parser")
        # Get the review data using its class
        # Scrape the required data from the provided URL
        example_column = soup.find_all(attrs={"class": "text show-more__control"})
        example_list = []
        for x in example_column[1:]:
            example_list.append(x.get_text())

        # Pass the scraped data to your fine-tuned BERT model for sentiment analysis
        predict_inputs = tokenizer(example_list, padding=True, truncation=True, return_tensors="tf")
        tf_output = best_model(**predict_inputs)

        # Get the sentiment predictions
        tf_output = tf_output[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        labels = ['Negative', 'Positive']  # (0:negative, 1:positive)
        label = tf.argmax(tf_prediction, axis=1)
        label = label.numpy()
        average_sentiment_score = np.mean(label)
        positive_count = sum(label)
        negative_count = len(label) - positive_count

        # Calculate the mean sentiment value from the predictions
        return render_template('result.html', sentiment_result=average_sentiment_score, positive_count=positive_count, negative_count=negative_count, url=url)
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Exception occurred: {str(e)}")
        # Handle the exception gracefully by displaying an error message
        return render_template('error.html', error_message='An error occurred. Please try again later.')


if __name__ == '__main__':
    app.run(debug=True)
