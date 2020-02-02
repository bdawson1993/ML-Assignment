import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import cv2


#function to load image
def LoadImg(tag, number):
    img = cv2.imread("train/" + str(tag) + "." + str(number) + ".jpg")
    img = cv2.resize(img, dsize=(100,100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img



img = LoadImg("cat", 50)
img2 = LoadImg("dog", 51)

newImg = np.concatenate((img, img2))

target = np.full(img.shape[0], 0)
target2 = np.concatenate((target, np.full(img2.shape[0], 1)))
model = neighbors.KNeighborsClassifier(5)
model.fit(newImg, target2)

testImg = LoadImg("dog",12)



predictions = model.predict(testImg)

print(predictions)

cv2.imshow("h", testImg)

plt.imshow(newImg, cmap=plt.get_cmap("gray"))
plt.show()

