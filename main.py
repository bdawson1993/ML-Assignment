import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from PIL import Image

img = Image.open("train/cat.0.jpg").convert("LA")

data = np.asarray(img)
print(data.shape)

classifiction = neighbors.KNeighborsClassifier(3)
classifiction.fit(data[:,:,0], np.full(data[:,:,0].shape,1))

predictions = classifiction.predict(data[:,:,0])

print(predictions[:,:])


