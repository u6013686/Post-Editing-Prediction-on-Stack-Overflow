import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from data_util import clean_str

def train_model(pattern,func):
    """
    This function trains and evaluates specified linear model for an edit type of data
    :param pattern: edit type
    :param func: linear function
    :return: None
    """
    # Load data from files
    positive_examples=[]
    negative_examples=[]
    with open('../data/' + pattern + "pos.txt", "r", encoding='latin-1') as f:
        for line in f:
            positive_examples.append(line.strip())
    with open('../data/' + pattern + "neg.txt", "r", encoding='latin-1') as f:
        for line in f:
            negative_examples.append(line.strip())
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    y = np.concatenate([[1 for _ in positive_examples], [0 for _ in negative_examples]], 0)
    f.close()

    # data preprocessing
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(x_text))

    # training
    batch_size= 200
    epoch = 10

    D = SGDClassifier(loss = func, warm_start=True)
    for ep in range(epoch):
        for batch in range(len(y)//batch_size):
            if batch == 0:
                tfidf1 = np.concatenate(
                    [tfidf[batch * batch_size:(batch + 1) * batch_size].toarray(), tfidf[-(batch + 1) * batch_size:].toarray()], 0)
                X_train1, X_test, y_train1, y_test = train_test_split(tfidf1, np.concatenate(
                    [y[batch * batch_size:(batch + 1) * batch_size], y[-(batch + 1) * batch_size:]], 0), test_size=0.1,
                                                                      random_state=42)
                D.partial_fit(X_train1, y_train1,classes=np.unique(y_train1))
            else:
                tfidf1 = np.concatenate([tfidf[batch * batch_size:(batch + 1) * batch_size].toarray(),
                                         tfidf[-(batch + 1) * batch_size:-batch * batch_size].toarray()],0)
                X_train, X_test, y_train, y_test = train_test_split(tfidf1,
                                        np.concatenate([y[batch * batch_size:(batch + 1) * batch_size],
                                        y[-(batch + 1) * batch_size:-batch * batch_size]],0), test_size=0.1, random_state=42)
                D.partial_fit(X_train, y_train)
            Dpred = D.predict(X_train1)
            k = 0
            for i in range(len(Dpred)):
                if (Dpred[i] >= 0.5 and y_train1[i] == 1) or (Dpred[i] < 0.5 and y_train1[i] == 0):
                    k += 1
            print ('epoch '+ str(ep + 1) + ': ' + str(int((float(batch) / (len(y)//batch_size)) * 100)) + '% accuracy: ' +
                   str(format(float(k)/len(X_train1), '.4f')))

            # calculate accuracy,precision and recall
            if batch % 10 == 0 and batch // 10 > 1:
                k = 0
                tp = 0
                fp = 0
                fn = 0
                test_size = 0
                Dpred = D.predict(X_test)
                test_size += len(X_test)
                for i in range(len(Dpred)):
                    if (Dpred[i] >= 0.5 and y_test[i] == 1) or (Dpred[i] < 0.5 and y_test[i] == 0):
                        k += 1
                    if Dpred[i] >= 0.5 and y_test[i] == 1: tp += 1
                    if Dpred[i] >= 0.5 and y_test[i] == 0: fp += 1
                    if Dpred[i] < 0.5 and y_test[i] == 1: fn += 1
                precision = format(float(tp) / (tp + fp), '.4f')
                recall = format(float(tp) / (tp + fn), '.4f')
                print('test accuracy:' + str(format(float(k) / test_size, '.4f')))
                print('test precision:' + str(precision))
                print('test recall:' + str(recall))

        k = 0
        tp = 0
        fp = 0
        fn = 0
        test_size = 0
        Dpred = D.predict(X_test)
        test_size += len(X_test)
        for i in range(len(Dpred)):
            if (Dpred[i] >= 0.5 and y_test[i] == 1) or (Dpred[i] < 0.5 and y_test[i] == 0):
                k += 1
            if Dpred[i] >= 0.5 and y_test[i] == 1: tp += 1
            if Dpred[i] >= 0.5 and y_test[i] == 0: fp += 1
            if Dpred[i] < 0.5 and y_test[i] == 1: fn += 1
        precision = format(float(tp) / (tp+fp), '.4f')
        recall = format(float(tp) / (tp+fn), '.4f')
        print('test accuracy:' + str(format(float(k) / test_size, '.4f')))
        print('test precision:' + str(precision))
        print('test recall:' + str(recall))


# train_model('content','hinge') #svm
# train_model('content','log') #logistic regression
# train_model('link','hinge') #svm
# train_model('link','log') #logistic regression
# train_model('format','hinge') #svm
# train_model('format','log') #logistic regression
# train_model('image','hinge') #svm
# train_model('image','log') #logistic regression
