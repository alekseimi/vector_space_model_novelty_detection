import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from vector_space_model import fit_predict
from sklearn.metrics import confusion_matrix


rng = 123

'''
Vector space model
'''


def anomaly_vector_space_model(input_data):
    print(fit_predict(input_data, combination_rule='mean', distance_rule='euclidean'))



'''
Novelty detection using the Isolation Forest method using the sklearn library and scored by accuracy. 
features_train = input_val[0]
features_test = input_val[1]
labels_train = input_val[2]
labels_test = input_val[3]
'''


def anomaly_isolation_forest(input_data):
    clf = IsolationForest(max_samples=1000,
                          random_state=rng,
                          contamination=0.15)
    clf.fit(input_data[0])
    y_pred_test = clf.predict(input_data[1])
    y_pred_outliers = clf.predict(input_data[2])
    spam_array = np.full((y_pred_outliers.shape[0], 1), -1, dtype=int).flatten()
    ham_array = np.full((y_pred_test.shape[0], 1), 1, dtype=int).flatten()

    actual_array = np.append(ham_array, spam_array)
    prediction_array = np.append(y_pred_test, y_pred_outliers)

    print('predictions svm', accuracy_score(ham_array, y_pred_test))
    tn, fp, fn, tp = confusion_matrix(actual_array, prediction_array).ravel()

    fpr = fp / (tn+fp)
    fnr = fn / (tp + fn)
    accuracy = accuracy_score(actual_array, prediction_array)

    result_list = [fpr, fnr, accuracy]
    return result_list

'''
Anomaly detection using the OneClassSVM method using the sklearn library and scored by accuracy. 
features_train = input_val[0]
features_test = input_val[1]
labels_train = input_val[2]
labels_test = input_val[3]
'''


def anomaly_svm(input_data):
    clf = OneClassSVM(kernel='linear')
    clf.fit(input_data[0])
    y_pred_test = clf.predict(input_data[1])
    y_pred_outliers = clf.predict(input_data[2])
    spam_array = np.full((y_pred_outliers.shape[0], 1), -1, dtype=int).flatten()
    ham_array = np.full((y_pred_test.shape[0], 1), 1, dtype=int).flatten()
    print('outliers svm', accuracy_score(spam_array, y_pred_outliers))


    '''
    Append spam_array to ham_array and y_pred_outliers to y_pred_test
    '''
    actual_array = np.append(ham_array, spam_array)
    prediction_array = np.append(y_pred_test, y_pred_outliers)

    print('predictions svm', accuracy_score(ham_array, y_pred_test))
    tn, fp, fn, tp = confusion_matrix(actual_array, prediction_array).ravel()

    fpr = fp / (tn+fp)
    fnr = fn / (tp + fn)
    accuracy = accuracy_score(actual_array, prediction_array)

    result_list = [fpr, fnr, accuracy]
    return result_list


    #test_set_spam = input_data[2]
'''
Classifies text using the Naive Bayes classifier.
features_train = input_val[0]
features_test = input_val[1]
labels_train = input_val[2]
labels_test = input_val[3]
'''


def classify_naive_bayes(input_val):
    mnb = MultinomialNB(alpha=0.2)
    mnb.fit(input_val[0], input_val[2])
    prediction = mnb.predict(input_val[1])
    predicted_probas = mnb.predict_proba(input_val[1])
    return [input_val[3], prediction, predicted_probas]

'''
Classifies text using the Support Vector Machine classifier.
features_train = input_val[0]
features_test = input_val[1]
labels_train = input_val[2]
labels_test = input_val[3]
'''


def classify_support_vector_machine(input_val):
    svm = SVC(probability=True)
    svm.fit(input_val[0], input_val[2])
    prediction = svm.predict(input_val[1])
    predicted_probas = svm.predict_proba(input_val[1])
    return [input_val[3], prediction, predicted_probas]
