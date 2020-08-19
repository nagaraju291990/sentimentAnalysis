

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

train = pd.read_csv('tamil_train.tsv',sep='\t')

train_original=train.copy()

test = pd.read_csv('tamil_dev.tsv',sep='\t')

test_original=test.copy()

def pre_process():
    text = re.sub(r'[^A-Za-z]','',text)
    return text

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
train['text'] = train['text'].apply(pre_process)
train["label"] = train["category"].map(map_data1)
test['category'] = test['category'].str.strip()
test['ctext'] = test['text'].apply(pre_process)
test["label"] = test["category"].map(map_data1)


#combine train and test
#combine = train
combine = train.append(test,ignore_index=True,sort=True)

#combine['category'] = combine['category'].str.strip()
#combine["label"] = combine["category"].map(map_data1)

print(combine)


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

features = tfidf.fit_transform(combine.text).toarray()
labels = combine.label
#print(features.shape)


#Naive Bayes Classifier: the one most suitable for word counts is the multinomial variant:

X_train, X_test, y_train, y_test = train_test_split(train['text'], train['label'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

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
for t in combine.text:
    predict = clf.predict(count_vect.transform([t]))#.tostring()#.decode("utf-8")
    tmp = predict[0]
    out_array.append(tmp)
    #print(t,predict)
    #print(clf.predict(count_vect.transform(["mohanlal sir - look ..... kiddo.."])))

#print((out_array[0]))

#print(clf.predict(count_vect.transform([" Waiting to see the real hero undaa"])))
combine['label'] = out_array
#print(combine)
submission = combine[['text', 'label']]
submission["category"] = submission["label"].map(map_data2)
submission.to_csv('tam_result.tsv', index=False, sep='\t')

