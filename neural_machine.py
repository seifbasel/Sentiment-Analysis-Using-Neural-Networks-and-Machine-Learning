import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from hebbian import Hebbian
from madaline import MADALINE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

# List of columns to remove
columns_to_remove = ['number of reviews','Title','Clothing ID', 'Age', 'Division Name', 'Department Name', 'Class Name']

# Drop the specified columns
data.drop(columns_to_remove, axis=1, inplace=True)

# Preprocessing the Review Text column
def preprocess_text(text):
    # Check if text is a string
    if isinstance(text, str):
        # Convert text to lowercase
        text = text.lower()

        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]

        stemmer = SnowballStemmer(language='english')  # Initialize Snowball stemmer
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]  # Stem tokens

        # Join tokens back into text
        processed_text = ' '.join(stemmed_tokens)
        return processed_text
    else:
        return ""  # Return an empty string if text is not a string

# Apply preprocessing to the Review Text column
data['Review Text'] = data['Review Text'].apply(preprocess_text)

# Define features (X) and target (y)
X = data['Review Text']
y = data['Rating'].apply(lambda x: 1 if x > 3 else 0)  # Convert ratings to binary sentiment labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=2000)  # You can adjust the max_features parameter as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# Train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Train a Stochastic Gradient Descent (SGD) model
sgd_model = SGDClassifier()
sgd_model.fit(X_train_tfidf, y_train)

# Train a Support Vector Machine (SVM) model
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)

# Train a K-Nearest Neighbors (KNN) model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_tfidf, y_train)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Train a Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)

# Train a Gradient Boosting classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_tfidf, y_train)

# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Train a perceptron model
perceptron_model = Perceptron()
perceptron_model.fit(X_train_tfidf, y_train)

# Train an MLP feedforward neural network
mlp_feedforward_model = MLPClassifier(hidden_layer_sizes=(200,100,50), max_iter=1000, solver='adam')
mlp_feedforward_model.fit(X_train_tfidf, y_train)

# Train an MLP with backpropagation
mlp_backpropagation_model = MLPClassifier(hidden_layer_sizes=(200,100,50), max_iter=1000)
mlp_backpropagation_model.fit(X_train_tfidf, y_train)

# Train an Adaline model
adaline_model = SGDClassifier(loss='perceptron', eta0=0.1, learning_rate='constant', penalty=None)
adaline_model.fit(X_train_tfidf, y_train)

# Train a MADALINE model
madaline_model = MADALINE(input_size=X_train_tfidf.shape[1])
madaline_model.train(X_train_tfidf.toarray(), y_train.to_numpy(), learning_rate=0.1, epochs=100)





# Make predictions on the test set
y_pred_sgd = sgd_model.predict(X_test_tfidf)
y_pred_svm = svm_model.predict(X_test_tfidf)
y_pred_knn = knn_model.predict(X_test_tfidf)
y_pred_rf = rf_model.predict(X_test_tfidf)
y_pred_dt = dt_model.predict(X_test_tfidf)
y_pred_gb = gb_model.predict(X_test_tfidf)
y_pred_nb = nb_model.predict(X_test_tfidf)
y_pred_perceptron = perceptron_model.predict(X_test_tfidf)
y_pred_mlp_feedforward = mlp_feedforward_model.predict(X_test_tfidf)
y_pred_mlp_backpropagation = mlp_backpropagation_model.predict(X_test_tfidf)
y_pred_adaline = adaline_model.predict(X_test_tfidf)
y_pred_madaline = madaline_model.predict(X_test_tfidf.toarray())
y_pred_logistic = logistic_model.predict(X_test_tfidf)


# Evaluate the models
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
accuracy_mlp_feedforward = accuracy_score(y_test, y_pred_mlp_feedforward)
accuracy_mlp_backpropagation = accuracy_score(y_test, y_pred_mlp_backpropagation)
accuracy_adaline = accuracy_score(y_test, y_pred_adaline)
accuracy_madaline = accuracy_score(y_test, y_pred_madaline)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)


# Confusion Matrix 

print("\nLogistic Regression Model:")
print("Accuracy:", int(accuracy_logistic * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))

print("\nSGD Model:")
print("Accuracy:", int(accuracy_sgd * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_sgd))

print("\nSVM Model:")
print("Accuracy:", int(accuracy_svm * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

print("\nk-NN Model:")
print("Accuracy:", int(accuracy_knn * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

print("\nRandom Forest Model:")
print("Accuracy:", int(accuracy_rf * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nDecision Tree Model:")
print("Accuracy:", int(accuracy_dt * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nGradient Boosting Model:")
print("Accuracy:", int(accuracy_gb * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))

print("\nNaive Bayes Model:")
print("Accuracy:", int(accuracy_nb * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

print("\nPerceptron Model:")
print("Accuracy:", int(accuracy_perceptron * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_perceptron))

print("\nMLP Feedforward Model:")
print("Accuracy:", int(accuracy_mlp_feedforward * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mlp_feedforward))

print("\nMLP Backpropagation Model:")
print("Accuracy:", int(accuracy_mlp_backpropagation * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mlp_backpropagation))

print("\nAdaline Model:")
print("Accuracy:", int(accuracy_adaline * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_adaline))

print("\nMADALINE Model:")
print("Accuracy:", int(accuracy_madaline * 100), '%')
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_madaline))








# Visualize Confusion Matrices
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')


models = ["Perceptron", "MLP Feedforward", "MLP Backpropagation", "Adaline","Madaline","svm","Logistic Regression","SGD","knn","Random Forest Model","Decision Tree","Gradient Boosting","Naive Bayes"]
for i, y_pred in enumerate([ y_pred_perceptron, y_pred_mlp_feedforward, y_pred_mlp_backpropagation, y_pred_adaline,y_pred_madaline,y_pred_svm,y_pred_logistic,y_pred_sgd,y_pred_knn,y_pred_rf,y_pred_dt,y_pred_gb,y_pred_nb]):
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels=["Negative", "Positive"])
    plt.title(f"Confusion Matrix - {models[i]} Model")
    plt.show()


new_reviews = [
    "This dress is amazing I love it",
    "very good quality",
    "good fit",
    "nice product",
    "loved it",
    "The quality is very poor.",
    "the fit is bad",
    "i dont like it",
    "not worth money",
    "bad quality",
]

# Print the predictions using all models

print("\nPredictions using Perceptron Model:")
for review, prediction in zip(new_reviews, y_pred_perceptron):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()


print("\nPredictions using MLP Feedforward Model:")
for review, prediction in zip(new_reviews, y_pred_mlp_feedforward):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()


print("\nPredictions using MLP Backpropagation Model:")
for review, prediction in zip(new_reviews, y_pred_mlp_backpropagation):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()    


print("\nPredictions using Adaline Model:")
for review, prediction in zip(new_reviews, y_pred_adaline):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()
    
print("\nPredictions using Madaline Model:")
for review, prediction in zip(new_reviews, y_pred_madaline):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()

print("\nPredictions using knn Model:")
for review, prediction in zip(new_reviews, y_pred_knn):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()

print("\nPredictions using svm Model:")
for review, prediction in zip(new_reviews, y_pred_svm):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()

print("\nPredictions using decestion tree Model:")
for review, prediction in zip(new_reviews, y_pred_dt):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()

print("\nPredictions using Logistic Regression Model:")
for review, prediction in zip(new_reviews, y_pred_logistic):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()

print("\nPredictions using random forest  Model:")
for review, prediction in zip(new_reviews, y_pred_rf):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()


print("\nPredictions using SGDClassifier  Model:")
for review, prediction in zip(new_reviews, y_pred_sgd):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()


print("\nPredictions using GradientBoostingClassifier  Model:")
for review, prediction in zip(new_reviews, y_pred_gb):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Review:", review)
    print("Predicted Sentiment:", sentiment)
    print()
