# Mail Recipients Prediction using Text and Graph Learning Techniques

## Introduction

This project aims at predicting recipients of emails. Given an email body, this code outputs 10 recipients and rank them by decreasing order or relevance. A few methods were tested including KNN, TfIdf, to compute distances between multiple texts. Finally the best approach I found was to combine both the similarity between email bodies and the frequency of emails sent to recipients.

Note that using metadata such as the day/hour an email is written could improve the accuracy of this approach.

## Dataset

Sample of the Enron email dataset, consisting in 43613 emails for training, sent from December 1998 to November 2001 by 125 different senders, and 2362 emails for test, sent after November 2001 by the same senders. 

## Evaluation Metric

The evaluation metric used for this project is Mean Average @10 (MAP@10), which is a classification metric sensitive to the rank of predicted recipients. 

## Requirements
+ python 3
+ pandas
+ sklearn

## Run the model

Just run the following command:
```bash
python run.py
```


