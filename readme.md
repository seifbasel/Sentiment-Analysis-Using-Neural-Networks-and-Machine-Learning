# Sentiment Analysis on Women's Clothing E-Commerce Reviews

This project performs sentiment analysis on a dataset of women's clothing e-commerce reviews. It aims to classify reviews as positive or negative based on their text content. Various machine learning and neural network models are utilized for this task.

## Overview

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment or opinion expressed in a piece of text. In this project, we leverage machine learning and neural network techniques to analyze customer reviews of women's clothing products from an e-commerce platform.

## Dataset

The dataset used in this project is sourced from [Women's Clothing E-Commerce Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) dataset available on Kaggle. It contains reviews written by customers along with their ratings and additional information.

## Preprocessing

Before modeling, the review text undergoes preprocessing steps including lowercasing, tokenization, stop word removal, and stemming to prepare it for analysis.

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

## Usage

To run the project:

1. Install the required dependencies mentioned in `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) and place it in the project directory.

3. Run the `sentiment_analysis.py` script to preprocess the data, train the models, and evaluate their performance:
    ```
    python sentiment_analysis.py
    ```

## Example Predictions

After training, the models can make predictions on new reviews. Here are predictions on a few sample reviews:

- "nice product" - Predicted Sentiment: Positive
- "good quality" - Predicted Sentiment: Positive
- "very bad" - Predicted Sentiment: Negative
- "bad quality" - Predicted Sentiment: Negative

## Contributors

- [seif basel](https://github.com/seifbasel)

## License

This project is licensed under the [MIT License](LICENSE).
