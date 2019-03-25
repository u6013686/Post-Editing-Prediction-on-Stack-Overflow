from keras import backend as K
from keras.models import model_from_json
import numpy as np
from collections import defaultdict
import json
import math
from data_util import clean_str
import re

def load_model_n_vocabulary():
    """
    This function loads model and vocabulary for visualization
    :return: model, vocabulary and sequence length of data
    """
    print('Load model')
    # load json and create model
    json_file = open('models/imagemodel.json','r')
    loaded_model_text = json_file.read()
    loaded_model = model_from_json(loaded_model_text)
    json_file.close()
    # load weights into new model
    loaded_model.load_weights('models/imagemodel.hdf5')
    #load vocabulary
    print('Load vocabulary')
    with open('vocabulary/imagevoc.txt', 'r') as fv:
        vocabulary = json.load(fv)

    # get sequence length from json file
    json_file = open('models/imagemodel.json', 'r')
    loaded_model_json = json.load(json_file)
    sequence_length = int(loaded_model_json['config']['layers'][0]['config']['batch_input_shape'][1])
    return loaded_model, vocabulary, sequence_length

def load_visualized_post(sequence_length, vocabulary):
    """
    This function loads the example posts in different format
    :param sequence_length: sequence length
    :param vocabulary: vocabulary dictionary
    :return: x is the vector of padded post mapped from vocabulary,
            sentence refers to original posts in string format,
            new_sentence refers to tokenized post
    """
    sentence = open('visual_example/image.txt').read()
    new_sentence = clean_str(sentence.replace('\n', '').lower()).split(' ')
    num_padding = sequence_length - len(new_sentence)
    padding_word = "<PAD/>"
    padded_sentence = new_sentence + [padding_word] * num_padding
    x = [vocabulary[sent] for sent in padded_sentence]
    x = [np.array([x, ]), 1.]       # reshape x
    return x, sentence, new_sentence


def generate_array(outputs, x):
    """
    This function generates array for output of specific layer
    :param outputs: the output of specific layer
    :param x: post in vector format
    :return: numpy array of output of specific layer
    """
    inp = loaded_model.input # placeholder
    functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
    layer_outs = functor(x)
    return np.asarray(layer_outs[0][0])


def get_max_loc(layer_out, filters):
    """
    This function traces back to locate max values in each feature map
    :param layer_out: output of the convolutional layer
    :param filters: filter of CNN
    :return: the indices of max values
    """
    max_loc = []
    phrase_len = len(filters[:, 0, 0, 0])
    for i in range(512):
        max_l = np.argmax(layer_out[:, 0, i])
        max_loc.append((max_l, phrase_len))
    max_loc = sorted(max_loc)
    return max_loc

def calculate_prob(loaded_model, x):
    """
    This function calcuates probability of phrases that belongs to positive class
    :param loaded_model: CNN model
    :param x: post to be visualized
    :return: posibility_rank records probability of phrases that belongs to postive class
            and corresponding phrase indices in the original post in descending order
    """

    outputs1 = []
    outputs2 = []
    outputs3 = []
    outputs4 = []
    outputs5 = []
    #get outputs and weights of all layers
    for layer in loaded_model.layers:
        if layer.get_config()['name'] == u'conv2d_1':
            outputs1.append(layer.output)
            filters1 = np.asarray(layer.get_weights()[0])
            layer_out1 = generate_array(outputs1, x)
        elif layer.get_config()['name'] == u'conv2d_2':
            outputs2.append(layer.output)
            filters2 = np.asarray(layer.get_weights()[0])
            layer_out2 = generate_array(outputs2, x)
        elif layer.get_config()['name'] == u'conv2d_3':
            outputs3.append(layer.output)
            filters3 = np.asarray(layer.get_weights()[0])
            layer_out3 = generate_array(outputs3, x)
        elif layer.get_config()['name'] == u'flatten_1':
            outputs4.append(layer.output)
            layer_out4 = generate_array(outputs4, x)
        elif layer.get_config()['name'] == u'dense_1':
            filters5 = np.asarray(layer.get_weights()[0])  # evaluation function
            outputs5.append(layer.output)
            layer_out5 = generate_array(outputs5, x)

    if layer_out5[1] > 0.5: # determine whether the post need to be edited
        print ('The post lacks of an image.')
        d = defaultdict(list)
        #get the indices of largest value for each filter, there are three kind of filters with different sizes
        locs = get_max_loc(layer_out1, filters1) + get_max_loc(layer_out2, filters2) + get_max_loc(layer_out3, filters3)
        for r in range(len(locs)):
            d[locs[r]].append(r)
        posibility_rank = []
        for k, v in d.items():
            # calulated probability by using function that is similar to Softmax
            posibility = math.e ** sum(layer_out4[ind] * filters5[ind, 1] for ind in v) / (
                    math.e ** sum(layer_out4[ind] * filters5[ind, 1] for ind in v) + math.e ** sum(
                layer_out4[ind] * filters5[ind, 0] for ind in v))
            posibility_rank.append((posibility, [i for i in range(k[0], (k[0] + k[1] - 1))]))

        posibility_rank = sorted(posibility_rank, reverse=True)
        return posibility_rank
    else:
        print ('The post does not lack of an image.')
    return None

def clean_str_2(string):
    """
    The RegEx is used to  find the indices of special character and recreate the post.
    They are a little different from the RegEx for training data preprocessing
    :param string: text
    :return: preprocessed text
    """
    string = re.sub(r"[^A-Za-z0-9(),!?{}\'\`\[\]\r\n\t@%;~.*^$]", " ", string)
    string = re.sub(r"\^", " ^ ", string)
    string = re.sub(r"\$", " $ ", string)
    string = re.sub(r"{", " { ", string)
    string = re.sub(r"\*", " * ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"~", " ~ ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"}", " } ", string)
    string = re.sub(r"@", " @ ", string)
    string = re.sub(r"%", " % ", string)
    string = re.sub(r"\n", " \n ", string)
    string = re.sub(r"\r", " \r ", string)
    string = re.sub(r"\t", " \t ", string)
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
    return string.strip()

def calculate_averg_weights(locs, words):
    """
    This function calculates weights of each words by averaging the probability of all phrases containing the word
    :param locs: probability of phrases and their indices
    :param words: tokenized post
    :return: probability list for words in posts
    """
    d = defaultdict(list)
    for pair in locs:
        for loc in pair[1]:
            d[loc].append(pair[0])
    word_prob = [0.5] * len(words) # neutral words with probability 0.5
    for k, v in d.items():
        if k < len(words):
            word_prob[int(k)] = sum(v) / len(v)
    return word_prob

def normalize(word_prob):
    """
    This function normalizes probability of words for clearer visualization
    :param word_prob: list of word probability
    :return: normalized list of word probability
    """
    word_prob = [(e - min(word_prob)) / (max(word_prob) - min(word_prob)) for e in word_prob]
    return word_prob

def find_indices(tokenized_post, ind_of_char, word_prob):
    """
    This functions finds the indices of special characters in original posts
    :param tokenized_post: tokenized_post
    :param ind_of_char: index of special character and corresponding new special character
    :param word_prob: list of word probability
    :return: new tokenized post that convert some special characters into html format
            and list of word probability that includes probability of special characters.
    """
    for i in range(len(ind_of_char)):
        tokenized_post[ind_of_char[i][0]] = ind_of_char[i][1]
        word_prob.insert(ind_of_char[i][0], 0.0) # special character with probability 0.0
    return tokenized_post, word_prob

def recreate_post(word_prob, words, post):
    """
    This function recreates the post by insert special character which have been striped in previous
    data preprocessing and get the new list of word importance degree
    :param word_prob: list of word probability
    :param words: word tokens for preprocessed posts
    :param post: original posts
    :return:
    """
    tokenized_post = [i for i in clean_str_2(post).split(' ')]
    tokenized_post = list(filter(lambda a: a != '', tokenized_post))
    indices = [i for i, x in enumerate(words) if x == '']
    num = 0

    for i in indices:
        del word_prob[i - num]
        num += 1
    indices_of_char = []
    special_char = ['{', '}', '@', '%', ';', '~', '.', '*', '^', '$']
    for i in range(len(tokenized_post)):
        if any(e in tokenized_post[i] for e in special_char):
            indices_of_char.append((i, tokenized_post[i]))
        if '\n' in tokenized_post[i]:
            indices_of_char.append((i, '<br />\n')) #\n in html format
        if '\t' in tokenized_post[i]:
            indices_of_char.append((i, '&nbsp;&nbsp;&nbsp;&nbsp;')) #\t in html format
    recreated_post, new_word_prob = find_indices(tokenized_post, indices_of_char, word_prob)
    return recreated_post, new_word_prob

def save_visualization_results(recreated_post, new_word_prob):
    """
    This function saves the visualization results in html format, where the saturation of background
    color represents importance degree of words
    :param recreated_post: new post generated
    :param new_word_prob: new word probability including special characters
    :return: None
    """
    with open("imagevisualization.html", "w") as html_file:
        # difine the background color according to the probability list
        for word, alpha in zip(recreated_post, new_word_prob):
            html_file.write('<font style="background: rgba(255, 0, 0, %f)">%s</font>\n' % (alpha, word))


loaded_model, vocabulary, sequence_length = load_model_n_vocabulary()
x, sentence, new_sentence = load_visualized_post(sequence_length, vocabulary)
posibility_rank = calculate_prob(loaded_model,x)
word_prob = normalize(calculate_averg_weights(posibility_rank,new_sentence))
recreated_post, new_word_prob = recreate_post(word_prob,new_sentence, sentence)
save_visualization_results(recreated_post, new_word_prob)
print ('Visualization file have been saved in local directory')


