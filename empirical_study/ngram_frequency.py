from nltk.tokenize import word_tokenize
from collections import defaultdict
import operator
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def compute_frequency(n_gram):

    print ('Start calculating the frequency of ' + str(n_gram) + '-gram...')
    stop_words = set(stopwords.words('english')) # define stop word set
    lemmatiser = WordNetLemmatizer() # define lemmatiser

    frequency = defaultdict(int)
    with open('PostHistory.xml', 'r', encoding='latin-1') as f:
        for line in f:
            if ' Comment="' in line:
                comment = re.search(r' Comment="(.*?)"', line).group(1) # extract comments from post history
                if 'characters in body' not in comment and 'proposed by' not in line: # ignore meaningless comments
                    s = re.sub(r'\d+','',comment).lower() # strip number
                    word_tokens = word_tokenize(s)
                    filtered_tokens = []
                    for w in word_tokens:
                        if w not in stop_words and w not in "\'!?,.;:I%$&#@*(){}[]<>-..." and w!='n\'t' and \
                                w != 'quot' and w != '\'s'  and w != 'gt': # strip some special characters
                            w = lemmatiser.lemmatize(w, pos = "v")
                            filtered_tokens.append(w)
                    # counting
                    if n_gram == 1: # unigram
                        for w in filtered_tokens:
                            frequency[w] += 1
                    elif n_gram == 2: # bigram
                        if len(filtered_tokens) >= 2:
                            for u, v in zip(filtered_tokens[:-1],filtered_tokens[1:]):
                                frequency[u + ' ' + v] += 1
                    elif n_gram == 3: # trigram
                        if len(filtered_tokens) >= 3:
                            for u, v, w in zip(filtered_tokens[:-2],filtered_tokens[1:-1],filtered_tokens[2:]):
                                frequency[u + ' ' + v + ' ' + w] += 1
                    else:
                        print ('the value of n_gram must be between 1 - 3.')

    #display the frequency in descending order
    key_num = 0
    sorted_frequency = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    for key in sorted_frequency:
        key_num += 1
        if key_num <= 100:
            print (key)

compute_frequency(n_gram = 1)
compute_frequency(n_gram = 2)
compute_frequency(n_gram = 3)
