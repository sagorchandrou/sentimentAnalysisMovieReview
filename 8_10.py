import csv #working with csv file
import re  #working for text cleaning
from nltk.corpus import stopwords  #removing stop word #print(stopwords.words("english"))#>>> import nltk >>> nltk.download('stopwords')
import string   #------
import nltk     #Fow word tokenize  #>>> import nltk>>> nltk.download('punkt')
from nltk.stem import WordNetLemmatizer   #for remove am,is,are-->be, i.e, plays,playing,played-->play #>>> import nltk>>> nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()       #------
import string
from nltk.tokenize import word_tokenize

from nltk import ngrams

 
stopwords_english = stopwords.words('english')
 
# clean words, i.e. remove stopwords and punctuation and # filter out short tokens
def clean_words(words, stopwords_english):
    words_clean = []
    for word in words:
        if word not in stopwords_english and word not in string.punctuation: #and len(word) > 1:
           # if(word.isalnum()):
            words_clean.append(word)    
    return words_clean

def word_lemmatizer(text): 
    textN=[lemmatizer.lemmatize(word) for word in text]
    return textN


# feature extractor function for unigram
def bag_of_words(words):    
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary


# feature extractor function for ngrams (bigram)#our modified code
def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        #if(item[0] in important_words):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)    
    return words_dictionary

'''
#feature extractor function for ngrams (bigram)
def bag_of_ngrams(words, n=2):
    
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)    
    return words_dictionary
'''

# i.e. that extracts both unigram and bigrams features
def bag_of_all_words(words, n=2):
    words_clean = clean_words(words, stopwords_english)
    words_clean=word_lemmatizer(words_clean)
    words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)
    words_clean=word_lemmatizer(words_clean_for_bigrams)
 
    unigram_features = bag_of_words(words_clean)
    bigram_features = bag_of_ngrams(words_clean_for_bigrams)
 
    all_features = unigram_features.copy()
    all_features.update(bigram_features)
 
    return all_features


# cleaning words is find for unigrams
# but this can omit important words for bigrams
# for example, stopwords like very, over, under, so, etc. are important for bigrams
# we create a new stopwords list specifically for bigrams by omitting such important words
important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']
 
stopwords_english_for_bigrams = set(stopwords_english) - set(important_words)
 

print('section processing completed... OK')

pos_reviews = []
neg_reviews = []
with open('D:/finalworkToday/IMDB.csv',mode='r',errors='ignore',encoding="utf8") as csv_file1:
    reader = csv.reader(csv_file1, delimiter=',')
    count_val=0 
    for row in reader:
        modifiedText=(row[0].lower())
        if count_val==0:          #If section for skip the first row[0] value, i.e, here value is='review'
            count_val=1
            continue        
        modifiedText=modifiedText.replace("n't",' not')
        modifiedText=modifiedText.replace("'s",'')
        words=word_tokenize(modifiedText)   #review=0,sentiment=1 

        #bag_of_all_words(words)
        if(row[1]=='positive'):
            pos_reviews.append(words)
        else:
            neg_reviews.append(words)                 
csv_file1.close() 
    
print('section processing completed... OK')
#print(pos_reviews)
# positive reviews feature set
pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_all_words(words), 'pos'))
 
# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_all_words(words), 'neg'))
print('section processing completed... OK')

#print(pos_reviews)
# positive reviews feature set
pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_all_words(words), 'pos'))
 
# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_all_words(words), 'neg'))
print('section processing completed... OK')
print (len(pos_reviews_set), len(neg_reviews_set)) # Output: (1000, 1000)
 
# radomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result everytime we run the program
from random import shuffle 
shuffle(pos_reviews_set)
shuffle(neg_reviews_set)
 
test_set = pos_reviews_set[:12500] + neg_reviews_set[:12500]
te_label=[]
train_set = pos_reviews_set[12500:] + neg_reviews_set[12500:]
tr_label=[]
print(len(test_set),  len(train_set)) # Output: (400, 1600)
    
    from nltk import classify
from nltk import NaiveBayesClassifier
 
classifier = NaiveBayesClassifier.train(train_set)
 
accuracy = classify.accuracy(classifier, test_set)
print(accuracy) # Output: 0.8025

