import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import cv2



#function to load image
def LoadImg(tag, number):
    img = cv2.imread("train/" + str(tag) + "." + str(number) + ".jpg")
    img = cv2.resize(img, dsize=(100,100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#function to predict what the image holds
def predict(model, data):
    predDef = ["Cat", "Dog"]
    predictions = model.predict(data)
    counts = np.bincount(predictions)
    

    print(len(counts))
    if len(counts) >= 1:
        if(counts[0] > counts[1]):
            return predDef[0]
        else:
            return predDef[1]
    else:
        return predDef[0]

    return -1
    









img = LoadImg("cat", 50)
img2 = LoadImg("dog", 51)
newImg = np.concatenate((img, img2))

#build up target array
target = np.full(img.shape[0], 0)
target2 = np.concatenate((target, np.full(img2.shape[0], 1)))

#build model
model = neighbors.KNeighborsClassifier(5)
model.fit(newImg, target2)



#test data
a = input("Number")
testImg = LoadImg("cat",a)
x = predict(model, testImg)
print(x)



plt.imshow(testImg, cmap=plt.get_cmap("gray"))
plt.show()

