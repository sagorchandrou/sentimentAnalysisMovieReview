import csv #working with csv file
import re  #working for text cleaning
from nltk.corpus import stopwords  #removing stop word #print(stopwords.words("english"))#>>> import nltk >>> nltk.download('stopwords')
import nltk     #Fow word tokenize  #>>> import nltk>>> nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer #for remove suffixes, i.e, plays,playing,played-->play
englishStemmer=SnowballStemmer("english")      #------
from nltk.stem import WordNetLemmatizer   #for remove am,is,are-->be, i.e, plays,playing,played-->play #>>> import nltk>>> nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()       #------


def frequency_extract(text):
    word_withFrequency=[]
    vocab_text=set(text)
    for w in vocab_text:
        word_withFrequency.append([w,text.count(w)])
    return word_withFrequency 

def remove_noise_byFrequency(text,threshold):
    acurate_word=[]
    newFlist=frequency_extract(text)
    for eachList in newFlist:
        if(eachList[1]>threshold):
            acurate_word.append(eachList[0])
    return acurate_word
 
    
def remove_duplicacy(text):
    withOut_duplicate=list(set(text))
    return withOut_duplicate


def word_lemmatizer(text):
    lem_text=[lemmatizer.lemmatize(index) for index in text]
    return lem_text


def word_streaming(text):
    strimed_text=[]
    for word in text:
        strimed_text.append(englishStemmer.stem(word))  
    return strimed_text


def remove_stopword(text):
    stop_words = set(stopwords.words("english"))
    #sentence = text
    words = nltk.word_tokenize(text)
    without_stop_words = [word for word in words if not word in stop_words]
    # filter out short tokens
    without_stop_words = [word for word in without_stop_words if len(word) > 2]
    return without_stop_words


def text_cleaning(text):
    # Removing html tags
    TAG_RE = re.compile(r'<[^>]+>')
    sentence= TAG_RE.sub('', text)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return remove_stopword(sentence)


def write_fileOut(child1,child2):
    with open('C:/Users/HP/Desktop/finalworkToday/trainTest.csv','a') as fd:
        fd.write(child1+","+child2+"\n")
    fd.close()
        
with open('C:/Users/HP/Desktop/finalworkToday/IMDB.csv',mode='r',errors='ignore',encoding="utf8") as csv_file1:
    reader = csv.reader(csv_file1, delimiter=',')
    count_val=0 
    for row in reader:
        if count_val==0:          #If section for skip the first row[0] value, i.e, here value is='review'
            write_fileOut(row[0],row[1])
            count_val=1
            continue
        sentence=row[0].lower()   #review=0,sentiment=1    
        text_clean=text_cleaning(sentence)
        text_lemalized=word_lemmatizer(text_clean)
        withOutDuplicate=remove_duplicacy(text_lemalized)
        cleanedContent= ' '.join(withOutDuplicate)
        write_fileOut(cleanedContent,row[1])
        print(count_val,end='\r')
        count_val=count_val+1
csv_file1.close() 

print('trainTest created')


import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm




trainTestData = pd.read_csv('C:/Users/HP/Desktop/finalworkToday/trainTest.csv')

# train Data
trainData = trainTestData[:25000]

# test Data
testData = trainTestData[25000:]


# Create feature vectors
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
#tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
train_vectors = vectorizer.fit_transform(trainData['review'])
test_vectors = vectorizer.transform(testData['review'])



#Perform classification with SVM, kernel=linear,rbf
clf = svm.SVC(kernel='linear')
t0 = time.time()
clf.fit(train_vectors, trainData['sentiment'])
t1 = time.time()
prediction_linear = clf.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1


# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))

from sklearn import metrics
cm_test=metrics.confusion_matrix(testData['sentiment'], prediction_linear)
print('From Cinfusion Matrics:')
print(cm_test)
d1=cm_test[0]
d2=cm_test[1]
tn=d1[0];fn=d1[1];fp=d2[0];tp=d2[1];
p=fn+tp
n=tn+fp
#print("total actual positive points %d"%(p))
#print("total actual negative points %d"%(n))

precision=tp/(tp+fp)
print("Precision",precision)
recall = tp/p
print("Recall",recall)
f1 = 2*((precision*recall)/(precision+recall))
print("f1-score",f1)
print('accuracy=',(tp+tn)*100/(tp+tn+fp+fn),'%')