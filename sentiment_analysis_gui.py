import tkinter as tk
from tkinter import ttk, messagebox
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# Ensure the necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained models and the TF-IDF vectorizer
models = {
    "Logistic Regression": joblib.load('logistic_model.pkl'),
    "SGD": joblib.load('sgd_model.pkl'),
    "SVM": joblib.load('svm_model.pkl'),
    "KNN": joblib.load('knn_model.pkl'),
    "Random Forest": joblib.load('rf_model.pkl'),
    "Decision Tree": joblib.load('dt_model.pkl'),
    "Gradient Boosting": joblib.load('gb_model.pkl'),
    "Naive Bayes": joblib.load('nb_model.pkl'),
    "Perceptron": joblib.load('perceptron_model.pkl'),
    "MLP Feedforward": joblib.load('mlp_feedforward_model.pkl'),
    "MLP Backpropagation": joblib.load('mlp_backpropagation_model.pkl'),
    "Adaline": joblib.load('adaline_model.pkl'),
    "MADALINE": joblib.load('madaline_model.pkl')
}
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess the input text
def preprocess_input(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = SnowballStemmer(language='english')
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    processed_text = ' '.join(stemmed_tokens)
    return processed_text

# Function to predict the sentiment
def predict_sentiment():
    review = review_input.get()
    model_name = model_var.get()
    if review:
        processed_review = preprocess_input(review)
        review_tfidf = tfidf_vectorizer.transform([processed_review])
        model = models[model_name]
        if model_name == "MADALINE":
            prediction = model.predict(review_tfidf.toarray())[0]
        else:
            prediction = model.predict(review_tfidf)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        result_label.config(text=f"Predicted Sentiment: {sentiment}")
        if sentiment == "Positive":
            result_label.configure(style="Positive.TLabel")
        else:
            result_label.configure(style="Negative.TLabel")
    else:
        messagebox.showwarning("Input Error", "Please enter a review.")

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis Tool")
root.geometry("500x400")
root.resizable(False, False)

# Apply a style
style = ttk.Style()
style.theme_use('clam')

# Define styles
style.configure('TLabel', font=('Helvetica', 14))
style.configure('TEntry', font=('Helvetica', 14))
style.configure('TButton', font=('Helvetica', 14), background='#4CAF50', foreground='white')
style.configure('TCombobox', font=('Helvetica', 14))
style.configure('Positive.TLabel', font=('Helvetica', 16), foreground='green')
style.configure('Negative.TLabel', font=('Helvetica', 16), foreground='red')
style.map('TButton', background=[('active', '#45a049')])

# Create and place the widgets
frame = ttk.Frame(root, padding="20 20 20 20")
frame.pack(expand=True)

review_label = ttk.Label(frame, text="Enter your review:")
review_label.grid(column=0, row=0, pady=10, sticky='W')

review_input = ttk.Entry(frame, width=50)
review_input.grid(column=0, row=1, pady=5, sticky='W')

model_label = ttk.Label(frame, text="Select Model:")
model_label.grid(column=0, row=2, pady=10, sticky='W')

model_var = tk.StringVar(root)
model_var.set("Logistic Regression")  # default value

model_menu = ttk.Combobox(frame, textvariable=model_var, values=list(models.keys()), state='readonly')
model_menu.grid(column=0, row=3, pady=5, sticky='W')

predict_button = ttk.Button(frame, text="Predict Sentiment", command=predict_sentiment)
predict_button.grid(column=0, row=4, pady=10)

result_label = ttk.Label(frame, text="", font=("Helvetica", 16))
result_label.grid(column=0, row=5, pady=20)

# Start the GUI event loop
root.mainloop()
