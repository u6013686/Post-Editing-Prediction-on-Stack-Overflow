import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time
import csv
def train_model(pattern):
    """
    This is Attention_based LSTM. The function will yield accuracy, precision and recall at the end.
    :param pattern: edit type
    :return: None
    """
    num = 0
    noise_amp=[]         #an empty list to store the second column
    with open(pattern+'test.csv', 'r') as rf:
        reader = csv.reader(rf, delimiter=',')
        for row in reader:
            num += 1
            if num < 1000:
                noise_amp.append(','.join(row[1:]))
            else:
                break
    rf.close()

    #load data
    names = ["label", "content"]
    test_csv = pd.read_csv(pattern + "test.csv", names=names)
    train_csv = pd.read_csv(pattern + "train.csv", names=names)
    shuffle_csv = train_csv.sample(frac=1) #shuffle data
    x_train = pd.Series(shuffle_csv["content"])
    y_train = pd.Series(shuffle_csv["label"])
    pd.to_numeric(y_train,errors='ignore')
    x_test = pd.Series(test_csv["content"])
    y_test = pd.Series(test_csv["label"])
    pd.to_numeric(y_test,errors='ignore')
    train_size = x_train.shape[0]

    dev_size = 500
    train_size -= dev_size

    # hyperparameters
    MAX_DOCUMENT_LENGTH = 300
    EMBEDDING_SIZE = 128
    HIDDEN_SIZE = 256 #64
    ATTENTION_SIZE = 128
    lr = 1e-3
    BATCH_SIZE = 256
    KEEP_PROB = 0.5
    MAX_LABEL = 2

    # data preprocessing
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)
    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)

    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    graph = tf.Graph()
    with graph.as_default():
        #placeholder
        batch_x = tf.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH])
        batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
        keep_prob = tf.placeholder(tf.float32)

        # embedding
        embeddings_var = tf.Variable(tf.random_uniform([n_words, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_x)
        W = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=0.1))

        # LSTM
        rnn_outputs, _ = bi_rnn(BasicLSTMCell(HIDDEN_SIZE), BasicLSTMCell(HIDDEN_SIZE),
                                inputs=batch_embedded, dtype=tf.float32)
        # Add attention
        fw_outputs = rnn_outputs[0]
        bw_outputs = rnn_outputs[1]
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)
        alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, HIDDEN_SIZE]), tf.reshape(W, [-1, 1])))
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, MAX_DOCUMENT_LENGTH, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)

        #dropout layer
        drop = tf.nn.dropout(h_star, keep_prob)

        # fully connected layer
        W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, MAX_LABEL], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))
        y_hat = tf.nn.xw_plus_b(drop, W, b)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=batch_y))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # Accuracy metric
        y_hat_prox = tf.nn.softmax(y_hat)
        prediction = tf.argmax(y_hat_prox, 1)
        ground_truth = tf.argmax(batch_y, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, ground_truth), tf.float32))

    steps = int(train_size / BATCH_SIZE * 10) # 10 epochs
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        #generate labels
        train_labels = tf.one_hot(y_train, MAX_LABEL, 1, 0)
        test_labels = tf.one_hot(y_test, MAX_LABEL, 1, 0)
        y_train = train_labels.eval()
        y_test = test_labels.eval()

        #generate x and y for training data, validation data
        dev_x = x_train[: dev_size, :]
        dev_y = y_train[: dev_size, :]
        x_train = x_train[dev_size:, :]
        y_train = y_train[dev_size:, :]

        def batch_predict(x_test, y_test):
            """
            As there might be memory error, We apply batch prediction to the testing data.
            Precision, recall and accuracy will be printed
            """
            length = np.array([(len(list(filter(lambda a: a != x_test[0][-1], x_test[i]))) - 1) // 30 * 30 for i in
                               range(len(x_test))])
            ba = len(x_test) // 1000
            rest = len(x_test) % 1000

            from collections import defaultdict
            d = defaultdict(list)
            if ba != 0:
                for i in range(ba - 1):
                    pred, truth = sess.run([prediction, ground_truth],
                                           feed_dict={batch_x: x_test[i * 1000:(i + 1) * 1000, :],
                                                      batch_y: y_test[i * 1000:(i + 1) * 1000, :], keep_prob: 1})
                    # np.savetxt('pred/' +pattern + str(i) +'pred.txt', pred)
                    # np.savetxt('pred/' +pattern + str(i) +'truth.txt', truth)
                    ba_length = length[i * 1000:(i + 1) * 1000]
                    for k in range(len(ba_length)):
                        d[ba_length[k]].append((pred[k], truth[k]))
            pred, truth = sess.run([prediction, ground_truth],
                                   feed_dict={batch_x: x_test[ba * 1000:(ba * 1000) + rest, :],
                                              batch_y: y_test[ba * 1000:(ba * 1000) + rest, :], keep_prob: 1})
            # np.savetxt('pred/' + pattern + str(ba - 1) + 'pred.txt', pred)
            # np.savetxt('pred/' + pattern + str(ba - 1) + 'truth.txt', truth)
            ba_length = length[ba * 1000:(ba * 1000) + rest]
            for k in range(len(ba_length)):
                d[ba_length[k]].append((pred[k], truth[k]))

            tp = 0
            p1 = 0
            p2 = 0
            tn = 0
            p = 0
            for k, v in d.items():
                for i in v:
                    if (i[0] == i[1] and i[0] == 1):tp += 1
                    if (i[0] == i[1] and i[0] == 0):tn += 1
                    if (i[0] == 1):p1 += 1
                    if (i[1] == 1):p2 += 1
                    p += 1
            if p1 == 0:preci = 1
            else: preci = float(tp)/p1
            reca = float(tp)/p2
            accur = float(tp + tn)/p
            print('precision ' + str(preci) + ' recall ' + str(reca) + ' accuracy ' + str(accur))

        print("train the data")
        last_dev_acc = 0
        for step in range(steps):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = x_train[offset: offset + BATCH_SIZE, :]
            batch_label = y_train[offset: offset + BATCH_SIZE, :]

            fd = {batch_x: batch_data, batch_y: batch_label, keep_prob: KEEP_PROB}
            l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)

            if step % 100 == 0:
                print("Step %d: loss : %f   accuracy: %f %%" % (step, l, acc * 100))

            if step % 500 == 0:
                print("******************************\n")
                dev_loss, dev_acc, pred, truth = sess.run([loss, accuracy,prediction, ground_truth], feed_dict={batch_x: dev_x, batch_y: dev_y, keep_prob: 1})
                print("Dev set at Step %d: loss : %f   accuracy: %f %%\n" % (step, dev_loss, dev_acc * 100))
                #np.savetxt(pattern + 'pred.txt', pred) #save for observation
                #np.savetxt(pattern + 'truth.txt', truth)
                if dev_acc < last_dev_acc and acc*100 > 75:
                    break
                last_dev_acc = dev_acc
                print("******************************")

        print("start predicting: ")
        batch_predict(x_test, y_test)
