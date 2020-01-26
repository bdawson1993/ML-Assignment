import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.metrics import accuracy_score
from scipy import ndimage
from scipy import misc

from sklearn.datasets import load_iris

iris = load_iris()

print(iris)
irisData = iris.data

classfier = neighbors.KNeighborsClassifier(3)
model = classfier.fit(iris.data, iris.target)

print(model.predict(irisData[1:5]))

