
from argparse import ArgumentParser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import re
#import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

parser = ArgumentParser(description='Sentiment analysis on code mixed data for Tamil and  Malayalam\n\r'+
                        "How to Run?\n" +
                        "python3 " + sys.argv[0] + " -i=train.tsv" + " -t=test.tsv -l=mal/tam"
                        )
parser.add_argument("-i", "--input", dest="inputfile",
                    help="provide training file name in tsv format",required=True)
parser.add_argument("-t", "--test", dest="testfile",
                    help="provide test file name in tsv format",required=True)
parser.add_argument("-l", "--lang", dest="lang",
                    help="provide lang=mal/tam",required=False)
parser.add_argument("-d", "--db", dest="uniqdb",
                    help="provide unigram database file",required=False)
parser.add_argument("-d2", "--db2", dest="bigramdb",
                    help="provide bigram database file",required=False)

args = parser.parse_args()

inputfile = args.inputfile
testfile = args.testfile
lang = args.lang
uniqdb = args.uniqdb
bigramdb = args.bigramdb

fp = open(uniqdb, "r", encoding="utf-8")
lines = fp.read().split("\n")
fp.close()

my_dict = {}
for line in lines:
	if(line == ""):
		continue
	cols = line.split("\t")
	my_dict[cols[0]] = cols[1]

fp2 = open(bigramdb, "r", encoding="utf-8")
lines2 = fp2.read().split("\n")
fp2.close()


for line in lines2:
    if(line == ""):
        continue
    cols = line.split("\t")
    my_dict[cols[0]] = cols[1]


train = pd.read_csv(inputfile,sep='\t')

train_original=train.copy()

test = pd.read_csv(testfile,sep='\t')

test_original=test.copy()

#print(my_dict)
def pre_process(text):
    text = re.sub(r'[^A-Za-z0-9 ]','',text) #remove punctuations and numbers
    text = text.lower() #convert to lower case
    text = re.sub(r' +',' ',text) #remove multiple spaces
    text = text.strip() #remove trailing spaces
    words = text.split(" ")
    print("Before :"+ text)
    return_text = ''
    bigrams = []
    j = 0
    k =2
    #text = re.sub(r'(\w+)', lambda m: my_dict.get(m.group(0)), text)
    for  i in range(len(words)-1):
        x = ''
        x = ' '.join(words[j:k])
        #print(i,x,j)
        j = j + 1
        k = k + 1
        if x in my_dict:
            x = x + ":" + my_dict[x]
        else:
            x = x + ":" + "0_XXX"
        x = re.sub(r"_", ":", x, 1)
        bigrams.append(x)

    print("Bigrams",bigrams)
    i = 0
    index = 0
    flag = 0
    for i in range(len(bigrams)-1):
        cur_item, next_item = bigrams[i], bigrams[i+1]
        #print(cur_item, next_item)
        freq1 = int(cur_item.split(":")[1])
        freq2 = int(next_item.split(":")[1])
        #print(freq1, freq2)
        
        if(freq1 > freq2 or 1 ):
            index = 0
            flag = 0
            cur_bigram = cur_item.split(":")[0]
            #cur_bigram = re.sub(r' ', '_', cur_bigram, 1)
            return_text += cur_bigram + " {" + cur_item.split(":")[2] + '} '
            #print("Greater frequency",return_text)
        else:
            index = index + 1
            if(index == 2):
                index = 0
                flag = 1
                cur_item = bigrams[i-1]#, bigrams[i+1]
                cur_bigram = cur_item.split(":")[0]
                #cur_bigram = re.sub(r' ', '_', cur_bigram, 1)
                return_text += cur_bigram + " {" + cur_item.split(":")[2] + '} '
            elif (index == 1 and i > 0 and flag == 0):
                unigram = cur_item.split(":")[0].split(" ")[0]
                if( unigram in my_dict):
                    return_text += unigram + ' {' + my_dict[unigram].split("_")[1] + '} '
                else:
                    return_text += unigram + ' '
            else:
                next
            #print("Less frequency",return_text)



    print("After:" + return_text + "\n")
    return return_text

map_data1 = {
    'Positive' : '1',
    'Negative' :'0' ,
    'unknown_state': '2',
    'Mixed_feelings': '3',
    'not-malayalam': '4',
    'not-tamil' :'4',
    'not-Malayalam': '4',
    'not-Tamil' :'4'
    }
train['category'] = train['category'].str.strip()
train['newtext'] = train['text'].apply(pre_process)
train["label"] = train["category"].map(map_data1)
#test['category'] = test['category'].str.strip()
test['newtext'] = test['text'].apply(pre_process)
#test["label"] = test["category"].map(map_data1)


#exit()
#combine train and test
combine = train
#combine = train.append(test,ignore_index=True,sort=True)

#combine['category'] = combine['category'].str.strip()
#combine["label"] = combine["category"].map(map_data1)

print(combine["newtext"].head(10))
print(test)


"""Text Representation


The classifiers and learning algorithms can not directly process the text documents in their original form, as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length. Therefore, during the preprocessing step, the texts are converted to a more manageable representation.

One common approach for extracting features from text is to use the bag of words model: a model where for each document, a complaint narrative in our case, the presence (and often the frequency) of words is taken into consideration, but the order in which they occur is ignored.

Specifically, for each term in our dataset, we will calculate a measure called Term Frequency, Inverse Document Frequency, abbreviated to tf-idf. We will use sklearn.feature_extraction.text.TfidfVectorizer to calculate a tf-idf vector for each of consumer complaint narratives:

    sublinear_df is set to True to use a logarithmic form for frequency.
    min_df is the minimum numbers of documents a word must be present in to be kept.
    norm is set to l2, to ensure all our feature vectors have a euclidian norm of 1.
    ngram_range is set to (1, 2) to indicate that we want to consider both unigrams and bigrams.
    stop_words is set to "english" to remove all common pronouns ("a", "the", ...) to reduce the number of noisy features.
"""

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(train.newtext).toarray()
labels = train.label
print(features.shape)


#Naive Bayes Classifier: the one most suitable for word counts is the multinomial variant:

X_train, X_test, y_train, y_test = train_test_split(train['newtext'], train['label'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

map_data2 = {
    '1': 'Positive',
    '0': 'Negative',
    '2':'unknown_state',
    '3': 'Mixed_feelings',
    '4':'not-malayalam/tamil',
 #   4 : 'not-tamil'
    }
models = [
    #RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    #LinearSVC(),
    MultinomialNB()
    #LogisticRegression(random_state=0),
]
#accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
entries = []
CV = 5
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

#print(entries)
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df.groupby('model_name').accuracy.mean())


out_array = []
for t in test.newtext:
    predict = clf.predict(count_vect.transform([t]))#.tostring()#.decode("utf-8")
    tmp = predict[0]
    out_array.append(tmp)
    #print(t,predict)
    #print(clf.predict(count_vect.transform(["mohanlal sir - look ..... kiddo.."])))

#print((out_array[0]))

#print(clf.predict(count_vect.transform([" Waiting to see the real hero undaa"])))
test['label'] = out_array
#print(combine)
#submission = test[['text', 'label']]
test["category"] = test["label"].map(map_data2)
test.to_csv('result.tsv', index=False, sep='\t')

