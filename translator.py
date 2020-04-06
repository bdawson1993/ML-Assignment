# encoding: utf-8
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, GRU, TimeDistributed, Dropout
from keras.preprocessing.text import text_to_word_sequence
import numpy as np

class Translator:
    def __init__(self):
        self.__german = list()
        self.__english = list()
        
        self.__germanWords = dict([('none', -1)])
        self.__englishWords = dict([('none', -1)])
        self.__currentID = 0
        self.__reverseLookup = dict([(-1, 'none')]) #used for tranforming matrixes back to words faster than iteracting through both dics
       
    #give each unique word a ID
    def __Tokenize(self, line, language):
        words = text_to_word_sequence(line)
       
        for i in words:
            #check if is already in dictionary
            if i in language:
                continue
            else:
                language[i] = self.__currentID
                self.__reverseLookup[self.__currentID] = i
                self.__currentID += 1
                
        return language
    
    #transform a sentence into a matrix
    def __SentenceToMatrix(self, word):
        words = text_to_word_sequence(word)
        mat = []
        
        #iterate through words looking up the id in the dictnaires
        for i in range(len(words)):
            if(words[i] in self.__germanWords):
                mat.append(self.__germanWords[words[i]])
            else:
                mat.append(self.__englishWords[words[i]])
                
       

        return mat

    #transform a matrix into a sentence
    def __MatrixToSentence(self, matrix):
        sent = str()
        for i in matrix:
            sent += str(self.__reverseLookup[i]) + " "
        return sent
            
            
    #load all text files
    def LoadAllText(self, tag):
        loc = "translation/" + "ger-"+tag + ".txt" 
       
        textFile = open(loc, "r", encoding="latin-1")
        
        #read all lines
        lines = textFile.readlines()
        for i in  range(len(lines)):
            split = lines[i].split("\t")
            self.__german.append(split[0])
            self.__english.append(split[1])
            
            
            #tokenize words
            self.__germanWords = self.__Tokenize(self.__german[i], self.__germanWords)
            self.__englishWords = self.__Tokenize(self.__english[i], self.__englishWords)
            
        
        print(f"Unique German Words {len(self.__germanWords)}")
        print(f"Unique English Words {len(self.__englishWords)}")
            
        print("Test Matrix")
        mat = self.__SentenceToMatrix(self.__english[0])
        print(mat)
        print(self.__MatrixToSentence(mat))
            
       

    #build model
    def BuildModel(self, tag):
        #convert all sentences to a matrix
        englishMat = []
        germanMat = []
        
        #build english list
        englishMax = 0
        
        for i in range(len(self.__english)):
            englishMat.append(self.__SentenceToMatrix(self.__english[i]))
            
            #find max
            if len(englishMat[i]) > englishMax:
                englishMax = len(englishMat[i])
            print(f"English Max {englishMax}")
            
        #build french list
        for i in range(len(self.__germanWords)):
            germanMat.append(self.__SentenceToMatrix(self.__german[i]))
        
        englishMat = np.asarray(englishMat)
        germanMat = np.asarray(germanMat)
        
        print(f"English Mat Shape {englishMat.shape}")
        print(f"Gemrna Mat Shape {germanMat.shape}")
        
        print(englishMat[0])
        
        if tag == "RNN":
            model = Sequential()
            
            model.add(GRU(256, input_shape=englishMat.shape))
            model.add(TimeDistributed(Dense(1024, activation="relu")))
            model.add(Dropout(0.5))
            model.add(TimeDistributed(Dense(len(self.__germanWords), activation='softmax')))
           
            model.compile(optimizer="adam", loss='categorical_crossentropy')
            model.fit(englishMat, germanMat)
       


        
        


