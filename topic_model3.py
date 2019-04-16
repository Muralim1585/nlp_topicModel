# -*- coding: utf-8 -*-
"""
Topic Model

This file helps to combine python with Topic Model File.
"""

import re
import itertools
import gensim
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

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

            self.texts = gensim.utils.simple_preprocess(str(self.sents), deacc=True)  # deacc=True removes punctuations
            return self
    
    def remove_stopwords(self):
        print("removing stopwords...")
        stop_words = stopwords.words('english')
        stop_words.extend(['chapter',''])
        self.texts = [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in self.texts]
        self.texts = [word for word in self.texts if len(word)>0]
        return self

    def make_bitrigrams(text,minCount,thres):
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(text, min_count=minCount, threshold=thres) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[text], threshold=thres)  
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        bigrams = [bigram_mod[doc] for doc in text]
        trigrams = [trigram_mod[bigram_mod[doc]] for doc in text]
        return bigrams,trigrams
    
    def lemmatization(self, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        print("lemmatizing...")
        for word in self.texts:
            doc = nlp(" ".join(word)) 
            self.textsLemmatized = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        return self

class topicModelFeatExtract:
    
    def __init__(self,tokens):
        self.tokens = tokens
        self.spacyTag = []
        self.spacyToken = []
        self.df = pd.DataFrame()
        self.nltkBigram = []
        self.nltkTrigram = []
        
    def nltk_grams(self):
        check2 = " ".join(itertools.chain.from_iterable(self.tokens))
        nltkPairs = nltk.pos_tag(nltk.word_tokenize(check2))
        #print(nltkPairs[:5])
        self.nltkBigram = nltk.ngrams(nltkPairs,2)
        self.nltkTrigram = nltk.ngrams(nltkPairs,3)
        return self

    def spacy_tags(self):
        check2 = " ".join(itertools.chain.from_iterable(self.tokens))
        # post cleaning getting noun pharses and noun qualifier phrases
        for eachWord in nlp(check2):
            self.spacyTag.append(eachWord.pos_)
            self.spacyToken.append(eachWord.text)
        return self
        
    def getphrasesdf(self):
        phraseWord = ["{} {}".format(pair[0],pair[1]) for pair in zip(self.spacyToken,self.spacyToken[1:])]
        phraseTags = [self.spacyTag[i] +" "+self.spacyTag[i+1] for i in range(0,len(self.spacyTag)-1)]
        #print(phraseTags)
        #print(len(phraseTags))
        #print(len(phraseWord))
        self.df['spacyPOS']=phraseTags#[:-1]
        self.df['Word']=phraseWord
        return self
    
    def grams(gramlist,min,max):
        filler = []
        for i in range(min,max+1):
            grams = [gramlist[j:] for j in range(i)]
            #print(grams)
            #print([pair for pair in zip(*grams)])
            filler.append([" ".join(pair) for pair in zip(*grams)])
            #filler = [*itertools.chain.from_iterable(filler)]
        return filler
        
    def compiledf(self,min,max):
        xWord=topicModelFeatExtract.grams(self.spacyToken,min,max)
        xTag=topicModelFeatExtract.grams(self.spacyTag,min,max)
        
        self.df['Word']=[val for sublist in xWord for val in sublist]#[*itertools.chain.from_iterable(xWord)]
        self.df['spacyPOS']=[val for sublist in xTag for val in sublist]#[*itertools.chain.from_iterable(xTag)]
        return self
        
    
    
if __name__ == "__main__":
    scarletPimp = topicModel(r"F:\Python\Topic Model\SP.txt")
    scarletPimp = scarletPimp.store_data()
    scarletPimpbyChapter = scarletPimp.gutenberg_getBook()
    
    texts=[]
    for i,chapterName in enumerate(scarletPimpbyChapter):
        print("chapter {} is being processed".format(i+1))
        processed = topicModelPrep(chapterName)
        processed.sent_to_words()
        processed.remove_stopwords()
        print("\n")
        texts.append(processed.texts)
    
    df = pd.DataFrame()
    for i in range(len(texts)):    
        #print(texts)
        df2 = topicModelFeatExtract(texts[i]).spacy_tags()
        #dfbyChapter = df2.getphrasesdf().df
        dfbyChapter = df2.compiledf(1,3).df
        dfbyChapter['chapNum'] = i+1
        #filter for only noun or adjective based phrases
        r = '(NOUN|ADJ)+'
        dfbyChapter = dfbyChapter[dfbyChapter.spacyPOS.str.contains(r)]
        df = pd.concat([df,dfbyChapter],axis=0) 
        
        
        
        # unigram added to df
        # get noun phrases
        # do topic model
        # viz
        
# =============================================================================
# 
# criteria = {'spacyPOS':['ADJ NOUN','NOUN NOUN'], 'chapNum':[1]}
# 
# for k,v in criteria.items():
#     print(df[k].isin(v))
#     
#     print(k)
#     print(v)
# 
# def dffilter(*args,**kwargs):
#     #print(args)
#     filtered=args[0]
#     for k,v in kwargs.items():
#         #print(k)
#         #print(v)
#         #print(filtered)
#         filtered = filtered[filtered[k].isin(v)]
#         print(len(filtered))
#         #print(filtered.k.unique())
#     return filtered
# 
# def dffilter(*args,**kwargs):
#     filtered=args[0]
#     return (filtered[filtered[k].isin(v)] for k,v in kwargs.items())
#         
# dffilter(spacyPOS=['ADJ NOUN','NOUN NOUN'], chapNum=[1])
# dffilter(spacyPOS=['NOUN NOUN'], chapNum=[1])
# dffilter(spacyPOS=['ADJ NOUN','NOUN NOUN'])
# 
# dffilter(df,chapNum=[1],spacyPOS=['NOUN NOUN','ADJ NOUN'])
# dffilter(df, chapNum=[1])
# 
# 
# df.Word.isin(['paris september'])
# df.Word.isin(['said captain'])
# type(df.spacyPOS)
# type(df.Word)
# 
# dffilter(df,Word=['scarlet pimpernel'])
# 
# xx = '!=2'
# re.findall('[!><=]',xx)
# 
# =============================================================================
# =============================================================================
# #spacy bigram implementation    
# check = wordsbyChapter['words'][0]
# check2 = " ".join(itertools.chain.from_iterable(check))
# # post cleaning getting noun pharses and noun qualifier phrases
# nlp_check2 = nlp(check2)
# tags=[]
# tokens=[]
# for eachWord in nlp_check2:
#     tags.append(eachWord.pos_)
#     tokens.append(eachWord.text)
#     
# 
# def printtoken(tokens):
#     return ["{} {}".format(pair[0],pair[1]) for pair in zip(tokens,tokens[1:])]
# 
# printtoken(tokens)[:5]
#     
# set(tags)#get only noun phrases and adj phrases
# 
# checkUnpacked =  [w for list in check for w in list]
# zipped = list(zip(checkUnpacked,tags))
# 
# counter=0
# for items in zipped:
#     #print(items)
#     if counter <5:
#         
#         token,tag = items
#         print(tag)
#         if tag == "NOUN" :
#             #print(token)
#             print(token)
#     counter+=1
#         
# zipped2 = zipped*2    
# 
# ########################
# phrase = {}
# phraseTags = [t + " " + t2 for t2 in tags[1:] for t in tags]
# phraseTags = [tags[i] +" "+tags[i+1] for i in range(0,len(tags)-1)]
# phraseTags[-1]
# phraseWord = [check[i][0] +" "+check[i+1][0] for i in range(0,len(check)-1)]
# len(phraseTags[:-1])
# 
# phrase['Tags'] = phraseTags
# phrase['Word'] = phraseWord
# len(phraseTags)
# len(phraseWord)
# phrasedf = pd.DataFrame({'POS':phraseTags[:-1],'Word':phraseWord})
# phrasedf.to_csv('phrase.csv')
# 
# 
# #nltk bigram implementaiton
# #https://stackoverflow.com/questions/39241709/how-to-generate-bi-tri-grams-using-spacy-nltk
# def pairs(phrase):
#     '''
#     Iterate over pairs of JJ-NN.
#     '''
#     tagged = nltk.pos_tag(nltk.word_tokenize(phrase))
#     for ngram in ngramise(tagged):
#         print(ngram)
#         tokens, tags = zip(*ngram)
#         #print(tags)
#         #if tags == ('JJ', 'NN'):
#         yield tokens
# 
# def ngramise(sequence):
#     '''
#     Iterate over bigrams and 1,2-skip-grams.
#     '''
#     for bigram in nltk.ngrams(sequence, 2):
#         yield bigram
#     for trigram in nltk.ngrams(sequence, 3):
#         yield trigram[0], trigram[2]
#     
# nltk_grams = list(pairs(check2))
# 
# xx = nltk.pos_tag(nltk.word_tokenize(check2))
# 
# 
# xx2 = list(nltk.ngrams(xx,2))
# 
# =============================================================================

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
