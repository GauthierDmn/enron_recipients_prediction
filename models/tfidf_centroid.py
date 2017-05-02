from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from  collections import Counter
import operator


def predict(training_info_preprocessed,training, test_info_preprocessed, test):

    # calculate tfidf vectors of documents in training and test sets
    tf = TfidfVectorizer(min_df=2)
    tfidf_vector = tf.fit_transform(training_info_preprocessed['clean_body'])
    tfidf_vector_test = tf.transform(test_info_preprocessed['clean_body'])

    # convert training set to dictionary
    emails_ids_per_sender = {}
    for index, series in training_info_preprocessed.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1]
        # temp_dictionary[mail] = temp_dictionary.get(mail, 0) + val
        emails_ids_per_sender[sender] = emails_ids_per_sender.get(sender, [])
        emails_ids_per_sender[sender].append(ids)

    # create address book with frequency information for each user
    address_books_freq = {}

    for sender, ids in emails_ids_per_sender.items():
        recs_temp = []
        for my_id in ids:
            recipients = training_info_preprocessed[training_info_preprocessed['mid'] == int(my_id)][
                'recipients'].tolist()
            if recipients == []:  # meaning we suppressed the line during the preprocessing
                break
            else:
                recipients = recipients[0].split(' ')
                # keep only legitimate email addresses
                recipients = [rec for rec in recipients if '@' in rec]
                recs_temp.append(recipients)
        # flatten
        recs_temp = [elt for sublist in recs_temp for elt in sublist]
        # compute recipient counts
        rec_occ = dict(Counter(recs_temp))
        # order by frequency
        sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse=True)
        # save
        address_books_freq[sender] = sorted_rec_occ

    # create address book with tf-idf vector information for each recipient based on the emails he received
    address_books = {}

    for sender, ids in emails_ids_per_sender.items():
        recs_temp = []
        for my_id in ids:
            recipients = training_info_preprocessed[training_info_preprocessed['mid'] == int(my_id)][
                'recipients'].tolist()
            if recipients == []:  # meaning we suppressed the line during the preprocessing
                break
            else:
                recipients = recipients[0].split(' ')
                # keep only legitimate email addresses
                recipients = [rec for rec in recipients if '@' in rec]
                recipients = [rec for rec in recipients if '.' in rec]
                tfidf = tfidf_vector[training_info_preprocessed[training_info_preprocessed['mid'] == int(my_id)].index]

                for rec in recipients:
                    recs_temp.append((rec, tfidf))

        address_books[sender] = recs_temp

    # reduce tfidf vectors by key (sum of normalized tfidf vectors by recipient in the address book of a sender)
    for k, v in address_books.items():
        temp_dictionary = dict()
        for (mail, val) in v: #
            temp_dictionary[mail] = temp_dictionary.get(mail, 0) + val
        address_books[k] = [(key, normalize(val,norm='l2', axis=1)) for (key, val) in temp_dictionary.items()]

        # convert test set to dictionary
        emails_ids_per_sender_test = {}
        for index, series in test_info_preprocessed.iterrows():
            row = series.tolist()
            sender = row[0]
            ids = row[1]
            emails_ids_per_sender_test[sender] = emails_ids_per_sender_test.get(sender, [])
            emails_ids_per_sender_test[sender].append(ids)

    # will contain email ids, predictions for random baseline, and predictions for frequency baseline
    predictions_per_sender = {}

    # number of recipients to predict
    k = 10
    n_iter = 1

    for key, value in emails_ids_per_sender_test.items():
        print('Sender:', n_iter, '/', len(emails_ids_per_sender_test.keys()))
        sender = key
        # get IDs of the emails for which recipient prediction is needed
        ids_predict = value
        ids_predict = [int(my_id) for my_id in ids_predict]
        content_preds = []
        for id_predict in ids_predict:
            tfidf_test = tfidf_vector_test[
                test_info_preprocessed[test_info_preprocessed['mid'] == int(id_predict)].index]
            temp_list = []
            for (mail, val) in address_books[sender]:
                # print(tfidf_test.shape,val.shape)
                temp_list.append(float(linear_kernel(tfidf_test, val)))
            simi_indices = [i for i in sorted(range(len(temp_list)), key=lambda k: temp_list[k])[::-1]]
            preds = [address_books[sender][index][0] for index in simi_indices if
                        sender != address_books[sender][index][0]][0:k]

            if len(preds) != 10:
                for freq_index in range(len(address_books_freq[sender])):
                    if address_books_freq[sender][freq_index][0] not in preds:
                        preds.append(address_books_freq[sender][freq_index][0])
                    if len(preds) == 10:
                        break

            content_preds.append(preds)

        predictions_per_sender[sender] = [ids_predict, content_preds]
        n_iter += 1

    return predictions_per_sender