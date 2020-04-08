import matplotlib.pyplot as plt
import numpy as np

#ML libs
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


import cv2
import log
import threading
import concurrent.futures
import multiprocessing
import progressbar
from keras.utils import to_categorical
import tensorflow as tf


class classifier:
    def __init__(self, greyscale):
       
        
        self.__value = 0
        self.__lock = threading.Lock()
        self.__model = 0
        self.__greyScale = greyscale
        self.__wasNN = False

        #load imgs
        
        
        log.start("Loading Images")
        
        #load cats image
        cats = self.__loadAllImgs("cat", 50)
        catsA = np.asarray(cats)
        catsTarget = np.full(catsA.shape[0], 0)
        print("Cats Shape: " + str(catsA.shape))


        #load dog images
        dogs = self.__loadAllImgs("dog", 50)
        dogsA = np.asarray(dogs)
        dogsTarget = np.full(dogsA.shape[0],1)
        print("Dogs Shape: " + str(catsA.shape))
        
        
        #join arrays and print results
        self.__imgs = np.concatenate((catsA, dogsA))
        self.__target = np.concatenate((catsTarget, dogsTarget))
        print("Final Image Shape: " + str(self.__imgs.shape))
        print("Final Target Shape: " + str(self.__target.shape))
        
        print(self.__target)
        
        log.stop("Images Loaded")

        #build model
        #print(self.__imgs.shape)
        print()
       
    
    #factory function that builds the selected model
    def BuildModel(self, tag):
        
        log.start("Building Model")

        
        #K- Nearest Neighbour
        if(tag == 1):
            nsamples, nx, ny, nz = self.__imgs.shape
            reshapedA = self.__imgs.reshape((nsamples, (nx * ny)*nz ))
            
            
            self.__model = neighbors.KNeighborsClassifier(5)
            self.__model.fit(reshapedA, self.__target)

        #Gausian
        if(tag == 2):
            nsamples, nx, ny = self.__imgs.shape
            reshapedA = self.__imgs.reshape((nsamples, (nx * ny)*nz ))
            
            
            self.__model = naive_bayes.GaussianNB()
            self.__model.fit(reshapedA, self.__target)


        #Convultinal Neaural Network
        if(tag == 3):
            
            
            
            
            self.__wasNN = True
            #cnnImgs = self.__imgs.reshape(6000,100,100,1)
            #print("Shape")
            #print(cnnImgs.shape)
            #self.__imgs = np.insert(self.__imgs, 1, axis=4)
            #self.__imgs.insert
            #self.__target = self.__target.reshape(10)
            
            #data for the network
            rows = self.__imgs[0].shape[0]
            cols = self.__imgs[0].shape[1]
            self.__target = to_categorical(self.__target)
           
        
        
            self.__model = Sequential()
            self.__model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(rows,cols,3))) #input layer

            self.__model.add(Conv2D(32, kernel_size=3, activation='relu')) #hidden

            self.__model.add(Flatten())
            self.__model.add(Dense(2, activation="softmax")) #outpur 

            
    
            self.__model.compile(optimizer="adam", loss='categorical_crossentropy')
            self.__model.fit(self.__imgs, self.__target)



        log.stop("Model Built...")



    def __LoadImg(self,tag, number, imgType):
        
        
        if(imgType == "test"):
            number = str(int(number) + 3000)

        path = imgType + "/" + str(tag) + "." + str(number) + ".jpg"
        
        img = cv2.imread(path)
        img = cv2.resize(img, dsize=(250,250))
        
        
        if(self.__greyScale == "y"):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
     
        return img

    def __loadAllImgs(self,tag, amount):
        mat = []
        mat.append(self.__LoadImg(tag, 0, "train"))
       
        for i in progressbar.progressbar(range(1,amount)):
            img = self.__LoadImg(tag, i, "train")
           # mat = np.concatenate((mat, img))
            mat.append(img)
            

        return mat

    


    def testData(self,index, goal):
        
        #a = input("Number to Test: ")
        index += 1
        
        #load test img and resize
        testImg = []
        testImg.append(self.__LoadImg(goal,index, "test"))
        testImg = np.asarray(testImg)
        
        #resize
        if(self.__wasNN == False):
            nsamples, nx, ny, nz = testImg.shape
            testImg = testImg.reshape(nsamples, (nx*ny)*nz)
        
        
        #predice
        x = self.predict(testImg)
        #print(x)
        
        #print(x)
        
        #thread safe addition
        if(x == goal):
            with self.__lock:
                localCopy = self.__value
                localCopy += 1
                self.__value = localCopy
                
        #print("RAN!")

    def predict(self, data):
        #print("predict")
        predDef = ["cat", "dog"]
        predictions = self.__model.predict(data)
        
        
       # print("Predicted Label:")
       # print(predictions[0,1])
        #counts = np.bincount(predictions)
    
        if self.__wasNN == False:    
            return predDef[predictions[0]]
        else:
            if predictions[0,0] > predictions[0,1]:
                return predDef[0]
            else:
                return predDef[1]
    
    def GetCorrect(self):
        return self.__value