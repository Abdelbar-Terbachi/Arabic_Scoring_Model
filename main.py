import csv
import re
from collections import Counter
from math import sqrt

import nltk
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import linear_kernel
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

stopwords = set(stopwords.words("arabic"))


def UncommonWords(A, B):
    # count will contain all the word counts
    count = {}
    # insert words of string A to hash
    for word in A:
        count[word] = count.get(word, 0) + 1
    # insert words of string B to hash
    for word in B:
        count[word] = count.get(word, 0) + 1
    # return required list of words
    return [word for word in count if count[word] == 1]


def word(data):
    data = word_tokenize(data)
    filter = []
    IS = ISRIStemmer()
    for w in data:
        if w not in stopwords:
            stem = IS.stem(w)
            filter.append(stem)
    return filter


def word_net(word_list):
    synonyms = []
    for word in word_list:
        try:
            syn = wordnet.synsets(word, lang=('arb'))[0]
            result = [lemma.name() for lemma in syn.lemmas(lang='arb')]
            synonyms.append(result)
        except:
            synonyms.append(word)
    return synonyms


# word then word_net
def similerty(answer_word, test_word):
    score = 0
    for words in answer_word:
        for word_s in test_word:
            if words == word_s:
                score += 0.5
    return score


def find_similar(tfidf_matrix, index, top_n=5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index + 1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]


def data():
    corpus = []
    file = "answer.csv"
    with open(file, "r") as paper:
        corpus.append((paper.read()))
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words=stopwords)
    tfidf_matrix = tf.fit_transform([content for content in corpus])
    return tfidf_matrix


def write(answer, q_answer, score):
    # csv file name
    filename = "answer.csv"
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        rows = {"answer": answer, "q_answer": q_answer, "score": score}
        writer.writerow(rows.values())
        csvfile.close()


def doc(doc):
    documents = []
    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()
    for sen in range(0, len(doc)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(doc[sen]))
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    return documents


def extract_words(sentence):
    ignore_words = ['a']
    words = re.sub("[^\w]", " ", sentence).split()
    print('words:' + re)
    print('words:' + words)
    words_cleaned = [w.lower() for w in words if w not in stopwords]
    return words_cleaned


def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = extract_words(sentence)
        words.extend(w)
    words = sorted(list(set(words)))
    return words


def bagofwords(sentence):
    vectorizer = CountVectorizer(stop_words=stopwords)
    return vectorizer.fit_transform(sentence).toarray()


def SVM():
    Corpus = pd.read_csv("answer.csv", "ar")
    test_data = Corpus['answer']
    train_data = Corpus['q_answer']
    test = doc(test_data)
    train = doc(train_data)
    vocabulary_train = tokenize_sentences(train)
    vocabulary_test = tokenize_sentences(test)
    vectorizer = CountVectorizer(stop_words=stopwords)
    X_train_counts = vectorizer.fit_transform(train)
    transformer = TfidfTransformer()
    tfidf_matrix_train = transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(tfidf_matrix_train, test)
    text_clf_svm = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='hinge',
                                                       penalty='l2',
                                                       alpha=1e-3, n_iter=5, random_state=
                                                       42)), ])
    text_clf = text_clf_svm.fit(train, test)
    predicted_svm = text_clf_svm.predict(train)
    score = np.mean(predicted_svm == test)
    return str(score)


def word2vec(word):
    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c * c for c in cw.values()))
    # lw = sum(c * c for c in cw.values())/len(cw)
    print('cw:' + '%.2f' % lw)
    # return a tuple
    return cw, sw, lw


def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch] * v2[0][ch] for ch in common) / v1[2] / v2[2]
    # return sum(v1[0][ch] * v2[0][ch] for ch in common) /(sqrt(sum(c*c for c in v1))*sqrt(sum(c * c for c in v1)) )


stemmer = nltk.stem.isri.ISRIStemmer()
remove_punctuation_map \
    = dict((ord(char), None) for char in
           string.punctuation)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stopwords)


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return (tfidf * tfidf.T).A[0, 1]
