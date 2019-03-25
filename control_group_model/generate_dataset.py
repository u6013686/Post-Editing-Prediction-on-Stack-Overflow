import csv

import numpy as np
import json

def create_dataset_LSTM(pattern):
    """
    This function create data in right format for LSTM based on the training and testing data used by CNN
    :param pattern: edit type
    :return: None
    """

    print('load vocabulary')
    with open('../vocabulary/' + pattern + 'voc.txt', 'r') as fv:
        vocabulary = json.load(fv)
    inv_voc = {v: k for k, v in vocabulary.items()}

    print('load training data')
    x_train = np.loadtxt('../splitted_data/' + pattern + 'x_train.txt')
    y_train = np.loadtxt('../splitted_data/' + pattern + 'y_train.txt')

    print('load testing data')
    x_test = np.loadtxt('../splitted_data/' + pattern + 'x_test.txt')
    y_test = np.loadtxt('../splitted_data/' + pattern + 'y_test.txt')

    with open(pattern + 'train.csv', 'w', newline='') as f2:
        writer = csv.writer(f2)
        for i in range(len(x_train)):
            s = ' '.join(inv_voc[k] for k in x_train[i]).replace(' <PAD/>', '')
            writer.writerow([int(y_train[i, 1]), s])

    with open(pattern + 'test.csv', 'w', newline='') as f2:
        writer = csv.writer(f2)
        for i in range(len(x_test)):
            s = ' '.join(inv_voc[k] for k in x_test[i]).replace(' <PAD/>', '')
            writer.writerow([int(y_test[i, 1]), s])

def create_dataset_fasttext(pattern):
    """
        This function create data in right format for fasttext based on the training and testing data used by LSTM
        :param pattern: edit type
        :return: None
    """
    with open(pattern + 'train.csv','rb') as fo:
        with open(pattern + 'train.txt','w') as fi:
            for line in fo:
                fi.write(line.decode('utf-8')[2:].replace('\r\n','') + ' __label__' + line.decode('utf-8')[0] + '\n')

    with open(pattern + 'test.csv','rb') as fo:
        with open(pattern + 'test.txt','w') as fi:
            for line in fo:
                fi.write(line.decode('utf-8')[2:].replace('\r\n','') + ' __label__' + line.decode('utf-8')[0] + '\n')