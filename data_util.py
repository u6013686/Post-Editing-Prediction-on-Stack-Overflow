import numpy as np
import re
import itertools
from collections import Counter
import json


def clean_str(string):
    """
    This function is for data preprocessing using RegEx
    :param string: original text
    :return: preprocessed text
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\[\]]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{1,3}", " ", string)
    string = re.sub(r"\s{5,}", " ", string)
    string = string.replace('    ','<code_indent>')
    return string.strip().lower()

def load_data_and_labels(pattern):
    """
    This function loads and concatenates positive data and negative data, generate
    labels for the post data
    :param pattern: edit type
    :return: all data after preprocessed and their labels
    """
    # Load data from files
    positive_examples=[]
    negative_examples=[]
    with open("data/" + pattern + "pos.txt", "r", encoding='latin-1') as f:
        for line in f:
            positive_examples.append(line.strip())
    with open("data/" + pattern + "neg.txt", "r", encoding='latin-1') as f:
        for line in f:
            negative_examples.append(line.strip())

    text = positive_examples + negative_examples  #concatenate positive data and negative data
    text = [clean_str(sent) for sent in text]
    text = [s.split(" ") for s in text] # tokenization

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    This function pads the sentences to the maximum length of the dataset
    :param sentences: all texts in list format
    :param padding_word: padding identifier
    :return: padded texts in list format
    """

    sequence_length = max(len(x) for x in sentences) # maximum length of sentences
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence) # number of padding elements for the sentence
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(pattern, sentences):
    """
    This function builds a vocabulary dictionary and a inverse vocabulary dictionary mapping word and index
    :param pattern: edit type
    :param sentences: all texts in list format
    :return: vocabulary dictionary and  a inverse vocabulary dictionary
    """
    # build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # build dictionary mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # build dictionary mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # save the vocabulary
    with open('vocabulary/' + pattern + 'voc.txt','w') as fv:
        json.dump(vocabulary,fv)
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    This function generates numpy array for x and y
    :param sentences: all texts in list format
    :param labels: y in list format
    :param vocabulary: vocabulary dictionary
    :return: numpy array x and y
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(pattern):
    """
    This function loads and preprocessed data, generate and save corresponding vocabulary
    :param pattern: edit type
    :return: x, y, vocabulary dictionary and  a inverse vocabulary dictionary
    """
    sentences, labels = load_data_and_labels(pattern)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(pattern, sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


