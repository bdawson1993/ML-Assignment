# encoding: utf-8
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

class Translator:
    def __init__(self):
        self.__text = list()
       


    def LoadAllText(self, tag):
        loc = "translation/" + "ger-"+tag + ".txt" 
       
        textFile = open(loc, "r", encoding="utf-8")
        
        #read all lines
        lines = textFile.readlines()
        for i in lines:
            self.__text.append(i)

        print(self.__text[1].split("\t"))


    def BuildModel(self, tag):
        print("test")
       


        
        


