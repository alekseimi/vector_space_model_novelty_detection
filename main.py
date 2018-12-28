import csv
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
import vector_space_model as vsm


#nltk.download()
corpus_list = ['lingspam', 'enronspam']


def write_to_csv(csv_name, add_list):
    with open(csv_name, "a") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(add_list)


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

def split_data_anomaly(processed_data, split_type, k):
    if split_type == 'ham':
        split_train = split_type
        split_test = 'spam'
    else:
        split_train = split_type
        split_test = 'ham'
    features_train, features_test, labels_train, labels_test = \
        train_test_split(processed_data, df['class'], test_size=0.33, random_state=k)
    index_train = [i for i, x in enumerate(labels_train) if x == split_train]
    index_test_ham = [i for i, x in enumerate(labels_test) if x == split_train]
    index_test_spam = [i for i, x in enumerate(labels_test) if x == split_test]
    return [features_train[index_train], features_test[index_test_ham], features_test[index_test_spam]]


def k_fold_x_val(processed_data, split_type, k=2, distance_rule_c='euclidean', combination_rule_c='min'):
    master_list = []
    if split_type == 'ham':
        split_train = split_type
        split_test = 'spam'
    else:
        split_train = split_type
        split_test = 'ham'
    for fold in range(k):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(processed_data, df['class'], test_size=0.33, random_state=fold)
        index_train = [i for i, x in enumerate(labels_train) if x == split_train]
        index_test_ham = [i for i, x in enumerate(labels_test) if x == split_train]
        index_test_spam = [i for i, x in enumerate(labels_test) if x == split_test]
        anomaly_split = [features_train[index_train], features_test[index_test_ham], features_test[index_test_spam]]
        master_list.append(pr.fit_predict(anomaly_split, distance_rule=distance_rule_c,
                                          combination_rule=combination_rule_c))

    csv_name = distance_rule_c + '_' + combination_rule_c + '_' + ' lingspam_spam'
    write_to_csv(csv_name+'.csv', master_list)
    return master_list





'''
Splits data into train and test sets for the purposes of classification
'''


def split_data_classifier(processed_data):
    features_train, features_test, labels_train, labels_test = \
        train_test_split(processed_data, df['class'], test_size=0.25, random_state=100)
    return [features_train, features_test, labels_train, labels_test]





#df['processed'] = df['text'].apply(text_preprocessor)
#df.to_csv('enronspam_processed.csv')


df = pd.read_csv("lingspam_processed.csv")
vectorized_output = vectorize_data(df)


#Classification
'''
print('Classification - Naive Bayes')
predict = pr.classify_naive_bayes(classifier_split)
print(accuracy_score(predict[0], predict[1]))
skplt.metrics.plot_roc(predict[0], predict[2])
plt.show()
skplt.metrics.plot_confusion_matrix(predict[0], predict[1])
plt.show()

print('Classification - Support Vector Machine')
predict = pr.classify_support_vector_machine(classifier_split)
print(accuracy_score(predict[0], predict[1]))
skplt.metrics.plot_roc(predict[0], predict[2])
plt.show()
skplt.metrics.plot_confusion_matrix(predict[0], predict[1])
plt.show()
'''





'''
 ANOMALY DETECTION 
    - One-Class SVM has the ability to capture the form of the dataset, which means it operates better on datasets
     which can be split into two (or more) distinct groups
    - Isolation Forest is based on the Random Forest algorithm, which makes is appropriate for multi-dimensional problems
    - Vector Space Model is based on the distance between datasets in the vector space
'''

#test = anomaly_split_ham[0].toarray()
list_isolation_forest = []
list_one_class_svm = []

for fold in range(10):
    anomaly_split = split_data_anomaly(vectorized_output, 'spam', fold)
    print(pr.anomaly_isolation_forest(anomaly_split))
    list_isolation_forest.append(pr.anomaly_svm(anomaly_split))
    list_one_class_svm.append(pr.anomaly_isolation_forest(anomaly_split))

write_to_csv('anomaly_svm_lingspam_spam.csv', list_isolation_forest)
write_to_csv('anomaly_isolation_forest_lingspam_spam.csv', list_one_class_svm)


print('Anomaly detection - One-Class SVM')


print('Anomaly detection - Isolation Forest')




#print('Vector space model')
#data_list = k_fold_x_val(vectorized_output, 'spam',
#                         combination_rule_c='max', distance_rule_c='euclidean', k=10)

