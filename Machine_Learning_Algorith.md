## N-Gram Analysis before converting into features
Before giving the input into TF-IDF and machine learning, data is pre-processed. That is cleaned, converted into an N-Gram Model. Also stop words, punctuations,numbers are also removed. Text is converte to lower case. This is our cleaned data.
Here we are using BiGram model. From the trained data, we extract bigrams and unigrams and map them with highest probabilistic(Frequency) category and save them as two seperate databases like unigram-frequency and bigram-frequency. 
Now the cleaned data would be matched against the unigram and bigram data. After processing the entire sentence/comment we would generate the following. 

#### For example:
**Initial text** :*trailer late ah parthavanga like podunga*
> Bigrams matched ['trailer late:002:Positive', 'late ah:007:Positive', 'ah parthavanga:002:Positive', 'parthavanga like:003:Positive',   'like podunga:155:Positive']
> **After:** *trailer late {Positive} late ah {Positive} ah parthavanga {Positive} parthavanga like {Positive}*
That is each sentence is converted into Bigram/Ngram model based on the match it gets and then trainined along with the category.

## Convert to features using TF-IDF(Term Frequency - Iverse Document Frequency)

> There are many techniques in Machine learning that can  be  used  to  categorize data. For machine learning the intital step is to convert the data(traninng or testing) to Numerals. These numerals can be called as feature vectors, i.e each vector has its own importance. Bag of Words or TF-IDF are the popular approaches for this purpose. For this purpose we used TF-IDF(Term Frequency- Inverse Document Frequency), after expermineting with Bag of words approach we observed that TF-IDF gives better results.  TF-IDF  is  a numerical  statistic  that  shows  the  relevance  or importance of  a word/keyword  to a document in a collection of corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.

**For example** there are comments on youtube videos that need to categorize for hate speech, spam and other which would require lot of manual effort if done manually. Here TF-IDF algorithm can be used to categorize the comments for the same.

### Term Frequency (TF) :

The first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document.
TF = number of times word appers in the document/total no of words in document
 
 Consider a document containing 100 words where the word dog appears 4 times.

> So, the Term Frequency (TF) for dog would be (4/100) = 0.04

### Inverse Document Frequency(IDF):

The second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
IDF = log(totoal no of documents/no of documents with word in it)

**For exmaple** we have 10000 documents where dog appears in 100 of these.
> then IDF =log(10000/100) = 2

> So TF-IDF = TF \* IDF = 0.04 \* 2= 0.08

### Split into Training and Validation sets:

Then split the data into training and validation set:
Then apply machine learning algorithm.
The problem here we are dealing is classification problem where we need to classify our text into *positive, negative, unknown_state, not malayalam, not tamil.*

> For this purpose we are using **Multinomial Naive Bayes Alogorithm**. 

**From wikipedia:**
> Naive Bayes Alogorithm is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features. 

We fit the trained data into the Multinomial NB model and train each sentence/comment with its given category.
Then we find the accuracy of the model using the validation set splitted ealier.
Then we find the categories of the test data using the above trained model.


(4) (PDF) Text Mining: Use of TF-IDF to Examine the Relevance of Words to Documents. Available from: https://www.researchgate.net/publication/326425709_Text_Mining_Use_of_TF-IDF_to_Examine_the_Relevance_of_Words_to_Documents [accessed Sep 15 2020].
