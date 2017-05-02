import argparse
import pandas as pd
from nltk.stem.porter import *
import string
import logging

STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
             'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
             'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
             'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
             'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


def clean_mail_regex(mail):

    mail = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", mail)
    mail = re.sub(r"\'s", " \'s", mail)
    mail = re.sub(r"\'ve", " \'ve", mail)
    mail = re.sub(r"n\'t", " n\'t", mail)
    mail = re.sub(r"\'re", " \'re", mail)
    mail = re.sub(r"\'d", " \'d", mail)
    mail = re.sub(r"\'ll", " \'ll", mail)
    mail = re.sub(r",", " , ", mail)
    mail = re.sub(r"!", " ! ", mail)
    mail = re.sub(r"\(", " ( ", mail)
    mail = re.sub(r"\)", " ) ", mail)
    mail = re.sub(r"\?", " ? ", mail)
    mail = re.sub(r"\s{2,}", " ", mail)

    punctuation = set(string.punctuation)
    mail = ''.join([w for w in mail.lower() if w not in punctuation])

    # Stopword removal
    mail = [w for w in mail.split() if w not in STOPWORDS]

    # Stemming
    stemmer = PorterStemmer()
    mail = [stemmer.stem(w) for w in mail]

    # Covenrt list of words to one string
    mail = ' '.join(w for w in mail)

    return mail


def preprocess(data_info, data):

    # convert training set to dictionary
    emails_ids_per_sender = {}
    for index, series in data.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        emails_ids_per_sender[sender] = ids

    sender = []
    for row in data_info['mid']:
        for key, value in emails_ids_per_sender.items():
            if str(row) in value:
                sender.append(key)
                break

    data_info['clean_body'] = data_info['body'].apply(clean_mail_regex)
    data_info['sender'] = sender
    data_info['new_date'] = data_info['date'].apply(lambda row: row[:10])
    data_info = data_info[data_info['new_date'].apply(lambda row: row[:3] == '200' or row[:3] == '199')]
    data_info['new_date'] = pd.to_datetime(data_info.new_date, format='%Y-%m-%d')
    data_info = data_info.sort_values('new_date').reset_index()

    #Remove empty mails
    empty_body_mid = data_info[data_info['clean_body'].isnull()]['mid'].tolist()
    data_clean = data_info[data_info['mid'].isin(empty_body_mid)==False].reset_index()

    if 'recipients' in data_clean.columns:
        return data_clean[['sender','mid','clean_body','recipients']]
    else:
        return data_clean[['sender','mid','clean_body']]


def load_datasets(path_training, path_training_info, path_test, path_test_info):

    training = pd.read_csv(path_training, sep=',', header=0)
    training_info = pd.read_csv(path_training_info, sep=',', header=0)
    test = pd.read_csv(path_test, sep=',', header=0)
    test_info = pd.read_csv(path_test_info, sep=',', header=0)

    return training, training_info, test, test_info



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess training_info csv')
    parser.add_argument('--path_training_info', type=str,
                        help='path of the training_info set')
    parser.add_argument('--path_test_info', type=str,
                        help='path of the test_info set')
    parser.add_argument('--path_training', type=str,
                        help='path of the training set')
    parser.add_argument('--path_test', type=str,
                        help='path of the test set')

    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    logger = logging.getLogger('preprocessing')
