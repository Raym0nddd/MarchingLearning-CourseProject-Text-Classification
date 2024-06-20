# -*- coding = utf-8 -*-
# @Time : 2024/6/7 22:38
# @Author : 王加炜
# @File : run_tradition.py
# @Software : PyCharm
# -*- coding = utf-8 -*-
# @Time : 2024/6/7 15:39
# @Author : 王加炜
# @File : preprocess.py
# @Software : PyCharmrun

import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
def load_data_and_labels(dataset):
    """Load data and labels."""
    data_path = os.path.join(dataset, 'data')
    texts = []
    labels = []
    class_list = [x.strip() for x in open(os.path.join(data_path, 'class.txt'), 'r', encoding='utf-8').readlines()]
    print(class_list)

    with open(os.path.join(data_path, 'train.txt'), 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            content, label = line.strip().split('\t')
            texts.append(content)
            labels.append(label)  # Convert labels to indices
    return texts, np.array(labels), class_list

def main():
    # Configurations: prompt the user to enter dataset and method
    dataset = input("Enter the dataset name: ")
    method = input("Enter the machine learning method (logistic, knn, bayes,tree,mlp): ")

    # Load data
    texts, labels, class_list = load_data_and_labels(dataset)

    # Text vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(texts)
    y = labels

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select and train the model based on user input
    if method == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif method == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif method == 'bayes':
        model = MultinomialNB()
    elif method == 'tree':
        model = DecisionTreeClassifier(random_state=42)
    elif method == 'mlp':
        model = MLPClassifier(random_state=42)
    else:
        raise ValueError("Unsupported machine learning method. Choose 'logistic', 'knn', 'bayes', 'tree', or 'mlp'.")

    model.fit(X_train, y_train)

    # Save the model to disk
    model_path = f'./{dataset}/saved_dict/{method}_model.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")

    # Load the model from disk
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    print("Model loaded successfully")

    # Predict and evaluate using the loaded model
    y_pred = loaded_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_list))

if __name__ == "__main__":
    main()
