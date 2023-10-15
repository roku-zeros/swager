from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

credit = []
with open('credit.txt', 'r', encoding='utf-8') as file:

    for line in file:

        cleaned_line = line.strip('. \n')

        credit.append(cleaned_line)

ipoteka = []
with open('ipoteka.txt', 'r', encoding='utf-8') as file:

    for line in file:

        cleaned_line = line.strip('. \n')

        ipoteka.append(cleaned_line)

card = []
with open('card.txt', 'r', encoding='utf-8') as file:

    for line in file:

        cleaned_line = line.strip('. \n')

        card.append(cleaned_line)

invest = []
with open('investm.txt', 'r', encoding='utf-8') as file:

    for line in file:

        cleaned_line = line.strip('. \n')

        invest.append(cleaned_line)


data = credit + card + invest + ipoteka
labels = [0] * 100 + [1] * 100 + [2] * 100 + [3]*100 # 0 for 'credit,' 1 for 'card,' 2 for 'invest', '3 for 'ipoteka'


tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


classifier = MultinomialNB()
classifier.fit(X_train, y_train)


def classify_input(input_message):
    input_vector = tfidf_vectorizer.transform([input_message])
    predicted_label = classifier.predict(input_vector)[0]

    categories = {0: 'credit', 1: 'card', 2: 'invest',3: 'ipoteka'}
    predicted_category = categories.get(predicted_label, 'unknown')  # Set as 'unknown' if not in categories

    return predicted_category


def run():
    input_message = "карта снять деньги"
    predicted_category = classify_input(input_message)
    print(f"The input message is related to: {predicted_category}")