import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import procedures as pr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk import FreqDist
import scikitplot as skplt
import re
import visual_space_model as vsm


#nltk.download()
corpus_list = ['lingspam', 'enronspam']


def visualize_token_frequency(tokens):
    token_frequency = FreqDist(tokens)
    for token, freq in token_frequency.most_common(20):
        print(token, freq)
    token_frequency.plot(20, cumulative=False)


def text_preprocessor(text):
    text = re.sub(r'\W+', ' ', text)
    print(text)
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stop_list = stopwords.words('english') + list(string.punctuation)
    tokens_no_stop = [stemmer.stem(token) for token in tokens if
                      token not in stop_list]
    no_integers = [token for token in tokens_no_stop if not token.isdigit()]
    return ' '.join(no_integers)


def vectorize_data(df):
    vectorizer = TfidfVectorizer()
    input_val = vectorizer.fit_transform(df['processed'].values.astype('U'))
    return input_val


'''
Splits data in the processed_data array according to the specified split type (spam or ham),
which serves as the normal (non-outlier) 
'''


def split_data_anomaly(processed_data, split_type):
    if split_type == 'ham':
        split_train = split_type
        split_test = 'spam'
    else:
        split_train = split_type
        split_test = 'ham'
    features_train, features_test, labels_train, labels_test = \
        train_test_split(processed_data, df['class'], test_size=0.33, random_state=100)
    index_train = [i for i, x in enumerate(labels_train) if x == split_train]
    index_test_ham = [i for i, x in enumerate(labels_test) if x == split_train]
    index_test_spam = [i for i, x in enumerate(labels_test) if x == split_test]
    return [features_train[index_train], features_test[index_test_ham], features_test[index_test_spam]]


'''
Splits data into train and test sets for the purposes of classification
'''


def split_data_classifier(processed_data):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(processed_data, df['class'], test_size=0.25, random_state=100)
    return [features_train, features_test, labels_train, labels_test]


df = pd.read_csv("enronspam_processed.csv")
print(df.head())
print(df.groupby(['class']).count())
print(df['class'])
#df['processed'] = df['text'].apply(text_preprocessor)
#df.to_csv('enronspam_processed.csv')
vectorized_output = vectorize_data(df)
classifier_split = split_data_classifier(vectorized_output)

predict = pr.classify_naive_bayes(classifier_split)
print(skplt.metrics.plot_roc(predict[0], predict[2]))
plt.show()


'''
#Classification
predict = pr.classify_naive_bayes(classifier_split)
print(accuracy_score(predict[0], predict[1]))
skplt.metrics.plot_roc(predict[0], predict[2])
plt.show()
skplt.metrics.plot_confusion_matrix(predict[0], predict[1])
plt.show()

predict = pr.classify_support_vector_machine(classifier_split)
print(accuracy_score(predict[0], predict[1]))
skplt.metrics.plot_roc(predict[0], predict[2])
plt.show()
skplt.metrics.plot_confusion_matrix(predict[0], predict[1])
plt.show()
'''


'''
 ZANAVANJE ANOMALIJ 
 V okviru eksperimenta bomo preizkušali dve različne metode zaznavanje anomalij oziroma novosti
    - Enorazredni SVM omogoča zajem oblike podatkovne množice, iz česar sledi, da deluje bolje,
    kadar lahko identificiramo dve dobro razdeljeni gruči
    - Isolation Forest algoritmi so snovani na algoritmu Random Forest, zaradi česar so bolje prilagojeni
    za večdimenzionalne probleme
'''

anomaly_split_ham = split_data_anomaly(vectorized_output, 'ham')
#test = anomaly_split_ham[0].toarray()
#pr.anomaly_svm(anomaly_split_ham)



#pr.anomaly_isolation_forest(anomaly_split_ham)
#pr.anomaly_svm(anomaly_split_ham)
#pr.anomaly_isolation_forest(anomaly_split_ham)
pr.fit_predict(anomaly_split_ham, combination_rule='min')

