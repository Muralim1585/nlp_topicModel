# -*- coding: utf-8 -*-
"""
Topic Model

This file helps to combine python with Topic Model File.
"""

import re
import itertools
import gensim
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import spacy



class topicModel:
    def __init__(self, path):
        self.path = path
        self.book = []
        self.chapter = []

    def print_data(self):
        with open(self.path, 'r') as f:
            for line in f:
                print(line)
                
    def store_data(self):
        with open(self.path, 'r') as f:
            
            for line in f:
                self.book.append(line)
            return self
        
    def book_getChapters(self):
        print("Breaking down by chapters...")
        #get chapter blocks
        chapBlockStart = [i for i, elem in enumerate(self.book) if re.search('(chapter)',elem.lower())]
        #print(len(self.book))
        chapBlockEnd = chapBlockStart[1:]
        chapBlockEnd.append(len(self.book))
        #chapBlockEnd = [chapBlockStart[1:],len(self.book)] #creates lists of lists : [[1,2],[10]]
        #print(chapBlockEnd)
        #chapBlockEnd = list(itertools.chain.from_iterable(chapBlockEnd)) #flattens [1,2,10]
        chapBlocks = zip(chapBlockStart,chapBlockEnd)
        
        for chapter in chapBlocks:
            #print(chapter)
            self.chapter.append(self.book[chapter[0]:chapter[1]])
        return self.chapter
        
    def gutenberg_getBook(self):
        indices = [i for i, elem in enumerate(self.book) if re.search(r'(START|END) OF THIS PROJECT GUTENBERG EBOOK THE SCARLET PIMPERNEL',elem)]
        #print(indices)
        self.book = self.book[indices[0]:indices[1]] #removes all info regarding the project
        return self.book_getChapters()
        
class topicModelPrep:
    
    def __init__(self, sents):
        self.sents = sents
        self.texts = []
        self.bigrams = []
        self.trigrams = []
        self.textsLemmatized = []
        print("preprocessing...")
        
    def sent_to_words(self):
        print("sents to words...")
        for sentence in self.sents:
            self.texts.append(gensim.utils.simple_preprocess(str(self.sents), deacc=True))  # deacc=True removes punctuations
            return self
    
    def remove_stopwords(self):
        print("removing stopwords...")
        stop_words = stopwords.words('english')
        stop_words.extend(['chapter',''])
        self.texts = [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in self.texts]
        return self

    def make_bitrigrams(self,minCount,thres):
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(self.texts, min_count=minCount, threshold=thres) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[self.texts], threshold=thres)  
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        self.bigrams = [bigram_mod[doc] for doc in self.texts]
        self.trigrams = [trigram_mod[bigram_mod[doc]] for doc in self.texts]
        return self
    
    def lemmatization(self, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        print("lemmatizing...")
        for word in self.texts:
            doc = nlp(" ".join(word)) 
            self.textsLemmatized.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return self
    
if __name__ == "__main__":
    scarletPimp = topicModel(r"F:\Python\Topic Model\SP.txt")
    scarletPimp = scarletPimp.store_data()
    scarletPimpbyChapter = scarletPimp.gutenberg_getBook()
    #spbyChapterPrep = list(topicModel.sent_to_words(scarletPimpbyChapter))
    processed = topicModelPrep(scarletPimpbyChapter)
    processed.sent_to_words()
    processed.remove_stopwords()
    
    
    
    
# =============================================================================
#
#     xx = pd.DataFrame(np.arange(len(scarletPimpbyChapter)),scarletPimpbyChapter,columns=['chapNo','chapters'])     
#     scarletPimp = topicModel.scarlet
#     scarletPimpbyChapter = scarletPimp.gutenberg_getBook()
#     print(books)
#     books.index()
#     indices = [i for i, elem in enumerate(books) if re.search(r'(START|END) OF THIS PROJECT GUTENBERG EBOOK THE SCARLET PIMPERNEL',elem)]
#     chapterwise = [i for i, elem in enumerate(books) if re.search('(chapter)',elem.lower())]
#     for chapters in chapterwise:
#         print(books[chapters])
#     books[59:383]
#     
#     chapters = zip(chapterwise,[chapterwise[1:],len(books)])
#     len(books)
#     for i in chapters:
#         print(i)
#         
#     books = books[indices[0]:indices[1]]
#     re.sub('\\n','',books[:-1])
#     
# =============================================================================
