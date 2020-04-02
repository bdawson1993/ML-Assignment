import numpy as np
import matplotlib.pyplot as plt
from images import Image 


#ML libs
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

from tensorflow import keras
from tensorflow.keras import layers

import cv2
import log
import threading
import concurrent.futures
import multiprocessing

class classifier:
    def __init__(self):
        self.__value = 0
        self.__lock = threading.Lock()
        self.__model = 0

        #load cats
        log.start("Loading Images")
        self.__cats = self.__loadAllImgs("cat", 3000)
        for i in range(len(self.__cats)):
            self.__cats[i].SetTarget(0)

        #load dogs
        self.__dogs = self.__loadAllImgs("dog", 3000)
        for i in range(len(self.__dogs)):
            self.__dogs[i].SetTarget(1)

        
        self.__imgs = self.__cats + self.__dogs
        log.stop("Images Loaded")

        #build model
        print()
       
    
    #factory function that builds the selected model
    def BuildModel(self, tag):
        log.start("Building Model")

        if(tag == 'KNN'):
            self.__model = neighbors.KNeighborsClassifier(5)
            self.__model.fit(self.__imgs, self.__target)

        if(tag == 'GNB'):
            self.__model = naive_bayes.GaussianNB()
            self.__model.fit(self.__imgs, self.__target)

        if(tag == 'CNN'):
            #self.__model = svm.SVC(kernel='poly', gamma=10)
            #self.__model.fit(self.__imgs, self.__target)
            #inputs = keras.Input(shape=(100,100,3), name='cat_image')
            #x = layers.Flatten(name='flattened_cat')(self.__cats)

            #x = layers.Dense(2, activation='relu', nam='encoder')(x)
            #x = layers.Dense(2, activation='relu', name='middle_layer')(x)
            #x = layers.Dense(len(self.__cats)/2, activation='relu', name='decoder')(x)
            print("test")
    log.stop("Model Built...")



    def __LoadImg(self,tag, number, imgType):
        if(imgType == "test"):
            number = str(int(number) + 3000)

        img = cv2.imread(imgType + "/" + str(tag) + "." + str(number) + ".jpg")
        img = cv2.resize(img, dsize=(100,100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        image = Image()
        image.SetData(img)


        return image

    def __loadAllImgs(self,tag, amount):
        mat = list()

        for i in range(1,3000):
            img = self.__LoadImg(tag, i, "train")
            mat.append(img)

        return mat

    def testData(self,index):
        #print(index)
        #a = input("Number to Test: ")
        testImg = self.__LoadImg("dog",index, "test")
        #print(testImg[1])
        x = self.predict(testImg)

            
        if(x == "Cat"):
            with self.__lock:
                localCopy = self.__value
                localCopy += 1
                self.__value = localCopy

    def predict(self, data):

        predDef = ["Cat", "Dog"]
        predictions = self.__model.predict(data)
        counts = np.bincount(predictions)
        
        if len(counts) > 1:
            if(counts[0] > counts[1]):
                return predDef[0]
            else:
                return predDef[1]
        else:
            return predDef[0]

        return -1

    def GetCorrect(self):
        return self.__value