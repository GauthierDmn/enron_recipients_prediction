import preprocessing
from models import tfidf_x_freq
import pandas as pd

#Load data
training = pd.read_csv('data/training_set.csv', sep=',', header=0)
training_info = pd.read_csv('data/training_info.csv', sep=',', header=0)
test = pd.read_csv('data/test_set.csv', sep=',', header=0)
test_info = pd.read_csv('data/test_info.csv', sep=',', header=0)

#We only take training sample from index 20000 after sorting by date
training_info_preprocessed = preprocessing.preprocess(training_info, training)[20000:].reset_index(drop=True)
test_info_preprocessed = preprocessing.preprocess(test_info, test)

preds = tfidf_x_freq.predict(training_info_preprocessed,test_info_preprocessed)

#Write the solution on a file
with open('predictions_final.txt', 'wb') as my_file:
    my_file.write(bytes(('mid,recipients' + '\n'),'utf-8'))
    for sender, preds in preds.items():
        ids = preds[0]
        rec_preds = preds[1]
        for index, my_preds in enumerate(rec_preds):
            my_file.write(bytes((str(ids[index]) + ',' + ' '.join(my_preds) + '\n'),'utf-8'))
