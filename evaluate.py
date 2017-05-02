import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split


def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def compute_mapk(dic_preds,X_test,k):

    Y_test = {}
    for sender,v in dic_preds.items():
        Y_test[sender] = [v[0],[]]
        for i in range(len(v[0])):
            Y_test[sender][1].append(X_test[X_test['mid']==v[0][i]]['recipients'].iloc[0].split(' ')[:10])

    Y_test_sorted = OrderedDict(sorted(Y_test.items()))
    predictions_per_sender_sorted = OrderedDict(sorted(dic_preds.items()))

    Y1 = []
    Y2 = []
    for (k1,v1),(k2,v2) in zip(predictions_per_sender_sorted.items(),Y_test_sorted.items()):
        Y1.extend(v1[1])
        Y2.extend(v2[1])
    return mapk(Y2,Y1,k)

def split(training_info_preprocessed,test_size=0.1):

    slice_index = 20000
    train_slice = training_info_preprocessed[slice_index:]

    X_train, X_test, y_train, y_test = train_test_split(
        train_slice[['sender', 'mid', 'clean_body', 'new_date', 'recipients']], train_slice['recipients'],
        test_size=test_size, random_state=42)
    X_train.reset_index(inplace=True)
    X_test.reset_index(inplace=True)

    return X_train, X_test