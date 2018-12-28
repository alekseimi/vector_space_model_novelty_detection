import numpy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import metrics


'''
Defines an interval of candidate thresholds by dividing the span between 
the upper and lower thresholds into ten equal parts
'''

period_a = 0.001


def draw_roc_curve(fpr, tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='Example')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def divide(lower_threshold, upper_threshold):
    period = (upper_threshold - lower_threshold)/9
    current_threshold = upper_threshold
    threshold_list = []
    i = 0
    while current_threshold >= lower_threshold:
        i = i + 1
        threshold_list.append(current_threshold)
        current_threshold = current_threshold - period
        if i == 9:
            if current_threshold - period < 0:
                current_threshold = lower_threshold
            else:
                current_threshold = current_threshold - period
            break

    threshold_list.append(current_threshold)
    return threshold_list


'''
The lowest threshold is defined as the highest possible value at which no spam message
is missclassified. 

distance_values - min, max or mean distance of the target email to the emails in the training set
starting_distance - min, max or mean of the distances in the training set
'''


def find_lower_threshold(distance_values, starting_distance):
    while True:
        ham_count = 0
        spam_count = 0
        for value in distance_values:
            if value >= starting_distance:
                spam_count = spam_count + 1
            else:
                ham_count = ham_count + 1
        if false_negative_ratio(ham_count, spam_count) == 0:
            break
        else:
            starting_distance = starting_distance - period_a
    return starting_distance


'''
The upper threshold is the lowest possible value at which no legitimate spam message was missclassified.

distance_values - min, max or mean distance of the target email to the emails in the training set
threshold - the lowest threshold as calculated by the function find_lower_threshold

'''


def find_upper_threshold(distance_values, threshold):
    while True:
        ham_count = 0
        spam_count = 0
        for value in distance_values:
            if value >= threshold:
                spam_count = spam_count + 1
            else:
                ham_count = ham_count + 1
        if false_positive_ratio(spam_count, ham_count) == 0:
            break
        else:
            threshold = threshold + period_a
    return threshold

'''
Fits and predicts the data utilizing the vector_space_model function based on the 'combination_rule' and 'distance_rule parameters'

combination_rule - 'mean', 'min' or 'max'
distance_rule - 'euclid' or 'manhattan'
'''


def fit_predict(input_data, combination_rule='mean', distance_rule='euclidean'):
    train_set = input_data[0]
    test_set_ham = input_data[1]
    test_set_spam = input_data[2]
    return vector_space_model(train_set, test_set_ham, test_set_spam, combination_rule, distance_rule)


'''
An implementation of a basic version of the anomaly-based spam detection filter as
proposed in the paper 'Study on the effectiveness of anomaly detection for spam filtering'
by Laorden et. al.
'''


def vector_space_model(train_set, test_ham, test_spam, combination_rule, distance_rule):
    distance_matrix = None
    distance_test_ham = None
    distance_test_spam = None
    if distance_rule == 'euclidean':
        distance_matrix = euclidean_distances(train_set, train_set)
        distance_test_ham = euclidean_distances(test_ham, train_set)
        distance_test_spam = euclidean_distances(test_spam, train_set)
    elif distance_rule == 'manhattan':
        distance_matrix = manhattan_distances(train_set, train_set)
        distance_test_ham = manhattan_distances(test_ham, train_set)
        distance_test_spam = manhattan_distances(test_spam, train_set)

    if combination_rule == 'mean':
        distance_matrix_mean = numpy.mean(distance_matrix[distance_matrix != 0])
        lower_threshold = find_lower_threshold(numpy.mean(distance_test_spam, axis=1), numpy.mean(distance_matrix_mean))
        upper_threshold = find_upper_threshold(numpy.mean(distance_test_ham, axis=1), lower_threshold)
        threshold_list = divide(lower_threshold, upper_threshold)
        return eval_thresholds(threshold_list, numpy.mean(distance_test_spam, axis=1), numpy.mean(distance_test_ham, axis=1))

    elif combination_rule == 'min':
        distance_matrix_min = numpy.min(distance_matrix[distance_matrix != 0])
        lower_threshold = find_lower_threshold(numpy.min(distance_test_spam, axis=1), distance_matrix_min)
        upper_threshold = find_upper_threshold(numpy.min(distance_test_ham, axis=1), lower_threshold)
        threshold_list = divide(lower_threshold, upper_threshold)
        return eval_thresholds(threshold_list, numpy.min(distance_test_spam, axis=1),
                              numpy.min(distance_test_ham, axis=1))
    elif combination_rule == 'max':
        distance_matrix_max = numpy.max(distance_matrix[distance_matrix != 0])
        lower_threshold = find_lower_threshold(numpy.max(distance_test_spam, axis=1), distance_matrix_max)
        upper_threshold = find_upper_threshold(numpy.max(distance_test_ham, axis=1), lower_threshold)
        threshold_list = divide(lower_threshold, upper_threshold)
        return eval_thresholds(threshold_list, numpy.max(distance_test_spam, axis=1),
                              numpy.max(distance_test_ham, axis=1))

'''
Evaluate each  thresholds fnr, fpr and wa
'''


def eval_thresholds(threshold_list, distance_test_spam, distance_test_ham):
    threshold_master = []
    for threshold in threshold_list:
        threshold_values_list = [threshold]
        ham_count = 0
        spam_count = 0
        for value in distance_test_spam:
            if value >= threshold:
                spam_count = spam_count + 1
            else:
                ham_count = ham_count + 1

        fnr = false_negative_ratio(ham_count, spam_count)
        threshold_values_list.append(fnr)
        ham_count = 0
        spam_count = 0
        for value in distance_test_ham:
            if value >= threshold:
                spam_count = spam_count + 1
            else:
                ham_count = ham_count + 1

        fpr = false_positive_ratio(spam_count, ham_count)

        threshold_values_list.append(fpr)
        wacc = weighted_accuracy(fnr, fpr)
        threshold_values_list.append(wacc)
        threshold_master.append(threshold_values_list)
    return threshold_master


def false_negative_ratio(fn, tp):
    return fn / (fn + tp)


def false_positive_ratio(fp, tn):
    return fp / (fp + tn)


def true_positive_ratio(tp, fp):
    return tp / (fp + tp)


def weighted_accuracy(fnr, fpr):
    return 1 - ((fnr + fpr) / 2)
