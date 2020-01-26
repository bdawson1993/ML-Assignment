import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.metrics import accuracy_score
from scipy import ndimage
from scipy import misc

##load image
f = misc.face()


classifier = neighbors.KNeighborsClassifier(3)
classifier.fit(f[:,:], f[:,:1])

predictions = classifier.predict(f[:,2])



