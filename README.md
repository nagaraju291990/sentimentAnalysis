# sentimentAnalysis
Sentiment analysis on Tamil and Malayalam code mixed data.

## Data
Training data, each sentence is marked as Positive/Negative/Mixed Feelings/Not-Malayalam,Not-Tamil/Unknown State

## How to run?
```
python3 main-ml.py -i=train.tsv -t=test.tsv 
python3 main-tam.py 
python3 main-mal.py
```

Result can be viewed in result-tam.csv and result-mal.csv respectively.

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
* Test data and training data file names are given already in the program itself.
* Map the labels like Negative, Positive, Unknown_state, Mixed_feelings, not-malayala/tamil to 0,1,2,3,4 repectively.
* Clean the data(not implemented yet)
* Convert the data into features using [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* Then these features are trained using [Multinomial NaiveBayes model](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) for from SKLEARN Module.
* From trained set we find the sentiment analysis of test data.
* We get values like 0,1,2,3,4 which will be mapped to original labels.
* Results can be found in result-tam.csv or result-mal.csv
