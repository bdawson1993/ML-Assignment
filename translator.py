# encoding: utf-8
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, GRU, TimeDistributed, Dropout,Bidirectional,Embedding,RepeatVector,LSTM
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import math

class Translator:
    def __init__(self):
        self.__german = list()
        self.__english = list()
        
        self.__germanWords = dict([('none', -1)])
        self.__englishWords = dict([('none', -1)])
        self.__currentID = 0
        self.__reverseLookup = dict([(-1, 'none')]) #used for tranforming matrixes back to words faster than iteracting through both dics
     
        
    def __convertList(self, data, max):
        
        npArray = np.empty([9998,max], dtype=int)
        
        for i in range(9998):
            for y in range(max):
                npArray[i,y] = data[i][y]
        
        #for i in range(11):
        #    print(npArray[0,i])
        
        return npArray
        
        
    
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
            
    def __padarray(self, orginal, maxV ):
         #pad so all values have same length
        for i in range(len(orginal)):
            
            if len(orginal[i]) < maxV:
                diff = maxV - len(orginal[i])
                
                #add values
                for y in range(0, diff):
                    orginal[i].append(-1)
                    
                #print(orginal[i]) #print new value
        orginal = np.asarray(orginal)
        return orginal

    #build model
    def BuildModel(self, tag):
        #pre process and clean data
        #convert all sentences to a matrix
        englishMat = []
        germanMat = []
        
        englishMax = 0
        germanMax = 0
        
        #build english list
        for i in range(len(self.__english)):
            englishMat.append(self.__SentenceToMatrix(self.__english[i]))
            
            #find max
            if len(englishMat[i]) > englishMax:
                englishMax = len(englishMat[i])
        
            
        #build german list
        for i in range(len(self.__german)):
            germanMat.append(self.__SentenceToMatrix(self.__german[i]))
            
            #find max
            if len(germanMat[i]) > germanMax:
                germanMax = len(germanMat[i])
                
                
        #pad both arrays
        englishMat = self.__padarray(englishMat, germanMax)
        germanMat = self.__padarray(germanMat, germanMax)
        
        
      
        
        
        
       
        
        #convert to numpy array
        #germanMat = np.asarray(germanMat)
        englishMat = self.__convertList(englishMat, germanMax)
        germanMat = self.__convertList(germanMat, germanMax)
       
        samples = englishMat.shape[0]
        words = englishMat.shape[1]
        
        #print(englishMat[0,0,0])
       # englishMat = englishMat.reshape((englishMat.shape[0],englishMat.shape[1], 1))
        #germanMat = germanMat.reshape((germanMat.shape[0], germanMat.shape[1],1))
        #NgermanMat = np.swapaxes(germanMat, 0, 1)
        
        print(f"English Mat Shape {englishMat.shape}")
        print(f"Gemrna Mat Shape {germanMat.shape}")
        print()
        print(f"English Max {englishMax}")
        print(f"German Max {germanMax}")
        
        print(f"rows {samples}")
        print(f"columns {words}")
        
        if tag == "RNN":
            model = Sequential()
            
            
           # model.add(Embedding(input_dim=5624, output_dim=64))
            #model.add(LSTM(128))
            #model.add(Dense(9998))
            
            #target = to_categorical(germanMat)
            
            model.add(Embedding(input_dim=len(self.__englishWords), input_length=12, output_dim=12)) #input
            model.add(GRU(12,input_shape=(samples, words), return_sequences=True)) # hidden 1
            model.add(Flatten())
            model.add(RepeatVector(12))
            
            model.add(Bidirectional(GRU(32, return_sequences=True)))
            #model.add(TimeDistributed((Dense(12, activation='relu'))))
            #model.add(Dropout(0.5))
            
            model.add(Flatten())
            model.add(Dense(12, activation='softmax')) #output
            
            print("Compiling")
            model.compile(optimizer="adam", loss='mean_squared_logarithmic_error')
            
            print("Fitting")
            model.fit(englishMat, germanMat, epochs=10)
            
            #englishMat.append(self.__SentenceToMatrix(self.__english[i]))
            
            y = []
            y.append(self.__SentenceToMatrix(self.__english[0]))
            y = self.__padarray(y, 12)
            
            x = model.predict(y)
            x = x.astype(int)
            print(x)
            
            val = []
            for i in range(12):
                #round was causing loss of data
                strV = float(x[0,i])
                print(strV)
                val.append(int(strV[0]))
            print()
            
            print("Inputed Sentence " + str(self.__english[0]))
            print("Predicted Output: " + str(self.__MatrixToSentence(val)))
                
                
       


        
        


