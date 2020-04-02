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

class classifier:
    def __init__(self):
        self.__value = 0
        self.__lock = threading.Lock()
        self.__model = 0

        #load imgs
        
        
        log.start("Loading Images")
        
        #load cats image
        cats = self.__loadAllImgs("cat", 3000)
        catsA = np.asarray(cats)
        catsTarget = np.full(catsA.shape[0], 0)
        print("Cats Shape: " + str(catsA.shape))


        #load dog images
        dogs = self.__loadAllImgs("dog", 3000)
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

        if(tag == 'KNN'):
            nsamples, nx, ny = self.__imgs.shape
            reshapedA = self.__imgs.reshape((nsamples, nx * ny ))
            
            
            self.__model = neighbors.KNeighborsClassifier(5)
            self.__model.fit(reshapedA, self.__target)

        if(tag == 'GNB'):
            self.__model = naive_bayes.GaussianNB()
            self.__model.fit(reshapedA, self.__target)

        if(tag == 'CNN'):
            self.__imgs = self.__imgs.reshape(6000,100,100,1)
            #self.__target = self.__target.reshape(10)

            
            model = Sequential()
            model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=self.__imgs.shape))

            model.add(Conv2D(32, kernel_size=3, activation='relu'))

            model.add(Flatten())
            model.add(Dense(10, activation="softmax"))

            
            model.compile(optimizer="adam", loss='categorical_crossentropy')
            print("Model fit")
            model.fit(self.__imgs, self.__target)



        log.stop("Model Built...")



    def __LoadImg(self,tag, number, imgType):
        if(imgType == "test"):
            number = str(int(number) + 3000)

        img = cv2.imread(imgType + "/" + str(tag) + "." + str(number) + ".jpg")
        img = cv2.resize(img, dsize=(250,250))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def __loadAllImgs(self,tag, amount):
        mat = []
        mat.append(self.__LoadImg(tag, 0, "train"))
       
        for i in progressbar.progressbar(range(1,3000)):
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
        nsamples, nx, ny = testImg.shape
        testImg = testImg.reshape(nsamples, nx*ny)
        
        
        #predice
        x = self.predict(testImg)
        #print(x)
        
        #print(x)
        
        #thread safe addition
        if(x.Trim() == goal):
            with self.__lock:
                localCopy = self.__value
                localCopy += 1
                self.__value = localCopy
                
        #print("RAN!")

    def predict(self, data):
        #print("predict")
        predDef = ["cat", "dog"]
        predictions = self.__model.predict(data)
        
        
        #print("Predicted Label:"+ str( predictions))
        #counts = np.bincount(predictions)
        
        return predDef[predictions[0]]

    def GetCorrect(self):
        return self.__value