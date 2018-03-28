from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
import numpy

# text
my_text = ["This is my text and I don't want to recognize different words as different terms: term and terms have same root, also house and houses. I don't wan't to recognize ''t' as term, neither punctuations. I wan't all Donald and Donalds be recognized with same root."]

# stop words
new_stop_words=["Donald"]
my_stop_words = text.ENGLISH_STOP_WORDS.union(numpy.concatenate([new_stop_words,list(punctuation)]))

# tokenizer
class MyTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        a = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        for i_term, term in enumerate(a, 0):
            if term=="Donalds":
                a[i_term] = "Donald"
        return a

vectorizer = TfidfVectorizer(tokenizer=MyTokenizer(), stop_words=my_stop_words)

x = vectorizer.fit(my_text)
words = vectorizer.get_feature_names()
print("words", words)