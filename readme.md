# Sentiment Analysis on Women's Clothing E-Commerce Reviews

This project performs sentiment analysis on a dataset of women's clothing e-commerce reviews. It aims to classify reviews as positive or negative based on their text content using various machine learning and neural network models.

## Overview

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment or opinion expressed in a piece of text. In this project, we leverage machine learning and neural network techniques to analyze customer reviews of women's clothing products from an e-commerce platform.

## Dataset

The dataset used in this project is sourced from the [Women's Clothing E-Commerce Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) dataset available on Kaggle. It contains reviews written by customers along with their ratings and additional information.

## Preprocessing

Before modeling, the review text undergoes preprocessing steps including:
- Lowercasing
- Tokenization
- Stop word removal
- Stemming

These steps prepare the text for analysis.

## Models

The following models are trained and evaluated for sentiment analysis:

- Logistic Regression
- Stochastic Gradient Descent (SGD)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- Decision Tree
- Gradient Boosting
- Naive Bayes
- Perceptron
- MLP Feedforward Neural Network
- MLP Backpropagation Neural Network
- Adaline
- MADALINE

## Evaluation

Each model's performance is evaluated using accuracy and confusion matrix metrics. Additionally, confusion matrices are visualized to provide a clear understanding of classification results.

## GUI Application

A GUI application is provided to input new reviews and predict their sentiment using the trained models. The GUI is built using Tkinter and ttkbootstrap for enhanced styling.

## Usage

### Prerequisites

1. Install the required dependencies mentioned in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) and place it in the project directory.

### Running the Project

1. Run the `sentiment_analysis.py` script to preprocess the data, train the models, and evaluate their performance:
    ```bash
    python sentiment_analysis.py
    ```

2. Run the `sentiment_analysis_gui.py` script to launch the GUI application:
    ```bash
    python sentiment_analysis_gui.py
    ```

### Example Predictions

After training, the models can make predictions on new reviews. Here are predictions on a few sample reviews:

- "nice product" - Predicted Sentiment: Positive
- "good quality" - Predicted Sentiment: Positive
- "very bad" - Predicted Sentiment: Negative
- "bad quality" - Predicted Sentiment: Negative

## Requirements

The project requires the following Python packages:

```plaintext
pandas
nltk
scikit-learn
joblib
matplotlib
seaborn
tk
ttkbootstrap
## Contributors

[text](https://github.com/seifbasel)
