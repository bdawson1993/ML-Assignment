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

        print(self.__text[1])


    def BuildModel(self, tag):

        #CNN
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32,1)))

        model.add(Conv2D(32, kernel_size=3, activation='relu'))

        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))

        model.compile(optimizer="adam", loss='categorical_crossentropy')


        
        


