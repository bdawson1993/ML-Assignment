import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import cv2
import log
import threading
import concurrent.futures
import multiprocessing


#function to load image
def LoadImg(tag, number, imgType):
    if(imgType == "test"):
        number = str(int(number) + 3000)

    img = cv2.imread(imgType + "/" + str(tag) + "." + str(number) + ".jpg")
    img = cv2.resize(img, dsize=(100,100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#function to predict what the image holds
def predict(model, data):
    predDef = ["Cat", "Dog"]
    predictions = model.predict(data)
    counts = np.bincount(predictions)
    
    if len(counts) > 1:
        if(counts[0] > counts[1]):
            return predDef[0]
        else:
            return predDef[1]
    else:
        return predDef[0]

    return -1
    

def loadAllImgs(tag, amount):
    mat = LoadImg(tag, 0, "train")

    for i in range(1,3000):
        img = LoadImg(tag, i, "train")
        mat = np.concatenate((mat, img))

    return mat

count = 0
def testData(index):
        print(index)
        #a = input("Number to Test: ")
        testImg = LoadImg("dog",index, "test")
        x = predict(model, testImg)

        if(x == "Cat"):
            count += 1

       # percentage = count / 500
        #print(f"Amount Correct {percentage}")

    #plt.imshow(testImg, cmap=plt.get_cmap("gray"))
    #plt.show()

threadCount = multiprocessing.cpu_count()

print("Loading Images...")
cats = loadAllImgs("cat", 3000)
catsTarget = np.full(cats.shape[0], 0)

dogs = loadAllImgs("dog", 3000)
imgs = np.concatenate((cats, dogs))
dogsTarget = np.full(dogs.shape[0],1)

target = np.concatenate((catsTarget, dogsTarget))
print("Images Loaded...")




#build model
print()
print("Building Model...")
log.start()
model = neighbors.KNeighborsClassifier(5)
model.fit(imgs, target)
print("Model Built...")
log.stop()


#test data
print("Testing...")

#spawn threads to go through

with concurrent.futures.ThreadPoolExecutor(max_workers=threadCount) as executor:
        executor.map(testData, range(500))


percentage = count / 500
print(f"amount corrent {percentage}")





