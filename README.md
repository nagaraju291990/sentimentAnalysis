# sentimentAnalysis
Sentiment analysis on Tamil and Malayalam code mixed data.

## Data
Training data, each sentence is marked as Positive/Negative/Mixed Feelings/Not-Malayalam,Not-Tamil/Unknown State

## How to run?
```
python3 main-ml-bigram.py -i=tamil_train.tsv -l=tam -d=tamil_uniq_freq.tsv -d2=tamil_bigram_freq.tsv -t=tamil_test\ -\ tamil_test.tsv
python3 main-ml.py -i=train.tsv -t=test.tsv
python3 main-tam.py 
python3 main-mal.py
```

Result can be viewed in result.csv respectively.

## Requirements
```
python3.6 and sklearn,pandas,numpy module
```

### To install python modules
```
pip3 install skealrn
pip3 install pandas
pip3 install numpy
```

## Algorithm

* Read data from tsv file.
* Input is training data, bigram data.
* Map the labels like Negative, Positive, Unknown_state, Mixed_feelings, not-malayala/tamil to 0,1,2,3,4 repectively.
* Clean/preprocess the data, it includes remove punctuations and numbers, convert to lower case, remove extra white spaces.
* Apply bigram analysis and unigram analysis on the data from bigram database.
* For ex: this is how a comment is processesed.  
    Before :trailer late ah parthavanga like podunga      
    Bigrams ['trailer late:002:Positive', 'late ah:007:Positive', 'ah parthavanga:002:Positive', 'parthavanga like:003 Positive',   'like podunga:155:Positive']  
    After:trailer late {Positive} late ah {Positive} ah parthavanga {Positive} parthavanga like {Positive}  
* Convert the data into features using [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* Then these features are trained using [Multinomial NaiveBayes model](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) for from SKLEARN Module.
* From trained set we find the sentiment analysis of test data.
* We get values like 0,1,2,3,4 which will be mapped to original labels.
* Results can be found in result.tsv file.
