from gensim import corpora, models
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser

print ('build corpus...')
text = []
stop_words = set(stopwords.words('english')) # define stop word set
lemmatiser = WordNetLemmatizer() # define lemmatiser

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
                            w != 'quot' and w != '\'s' and w != 'gt': # strip some special characters
                        w = lemmatiser.lemmatize(w, pos="v")
                        filtered_tokens.append(w)
                text.append(filtered_tokens)

print ('parse the text to generate bigrams and trigrams...')
phrases= Phrases(text, min_count=1, threshold=3) #parsing
trigram  = Phraser(phrases)
text = [trigram[sentence] for sentence in text]
dictionary = corpora.Dictionary(text) # generate vocabulary dictionary
corpus = [dictionary.doc2bow(text) for text in text] # build the corpus

print ('train the LDA model')
lda_model = models.LdaModel(corpus, num_topics = 8, id2word = dictionary, passes = 1) # training LDA model

for top in lda_model.print_topics(8): # print 8 topics
    print (top)
