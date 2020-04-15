from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import pandas as np
import matplotlib.pyplot as plt
import cv2
from fastai.imports import *
import tensorflow as tf
from tensorflow import keras
#Function to normalize series of images (photo_batch) into data and labels within a Category isSick(Pneumonia or healthy)
def get_test(all_data,all_labels,photo_batch,directory,isSick):
	i = 0;
	number_of_images = len([f for f in os.listdir(directory)if os.path.isfile(os.path.join(directory, f))])
	if isSick == 1:
		test = "Pneumonia"
	else:
		test = "healthy"
	for img in photo_batch:
		print("Getting image(",test ,"): ",i," of ",number_of_images, end="\r")
		img = cv2.imread(str(img))
		img = cv2.resize(img, (100,100))
		img = img.flatten()
		all_data.append([img])
		all_labels.append(isSick)
		i = i + 1
	print("Getting image(",test ,"): ",i," of ",number_of_images, end="\r")
	print("\n")
	pass
# Data path
data_directory = Path('data/chest_xray/')

# Test set path
test_directory = data_directory / 'test'

# Get the path to the sub-directories
test_normal_dir = test_directory / 'NORMAL'
test_pneumonia_dir = test_directory / 'PNEUMONIA'

# Get the list of all the images
test_normal_cases = test_normal_dir.glob('*.jpeg')
test_pneumonia_cases = test_pneumonia_dir.glob('*.jpeg')

test_data = []
test_labels = []


print("Test Images:")
get_test(test_data,test_labels,test_normal_cases,test_normal_dir,0)
get_test(test_data,test_labels,test_pneumonia_cases,test_pneumonia_dir,1)

test_data1 = np.array(test_data)[:,0]

# Data path
data_directory = Path('data/chest_xray/')

# Train set path
train_directory = data_directory / 'train'

# Get the path to the sub-directories
train_normal_dir = train_directory / 'NORMAL'
train_pneumonia_dir = train_directory / 'PNEUMONIA'

# Get the list of all the images
train_normal_cases = train_normal_dir.glob('*.jpeg')
train_pneumonia_cases = train_pneumonia_dir.glob('*.jpeg')


train_data = []
train_labels = []

print("Train Images:\n")
get_test(train_data,train_labels,train_normal_cases,train_normal_dir,0)
get_test(train_data,train_labels,train_pneumonia_cases,train_pneumonia_dir,1)

train_data1 = np.array(train_data)[:,0]


print(2,"-NN: ")
kNN = KNeighborsClassifier(n_neighbors = 2)
kNN.fit(train_data1,train_labels)
score = kNN.score(test_data1,test_labels)
print(score)

import numpy as np
import matplotlib.pyplot as plt
neighbors = np.arange(1,10)

accuracies1 = [0.7516025641025641, 0.7932692307692307, 0.7467948717948718, 0.7788461538461539, 0.7371794871794872, 0.7548076923076923, 0.7355769230769231, 0.75, 0.7307692307692307]
accuracies2 = [0.7580128205128205, 0.8044871794871795, 0.7516025641025641, 0.7772435897435898, 0.7339743589743589, 0.7564102564102564, 0.7371794871794872, 0.7435897435897436, 0.7307692307692307]

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, accuracies1, label='size: 150x150')
plt.plot(neighbors, accuracies2, label='size: 100x100')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

#let us get the predictions using the classifier we had fit above
y_pred = kNN.predict(X_test)

test_labels = np.array(test_labels).flatten()

#import confusion_matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(test_labels,y_pred)

pd.crosstab(test_labels, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

# accuracies2 (100x100) = [0.7580128205128205, 0.8044871794871795, 0.7516025641025641, 0.7772435897435898, 0.7339743589743589, 0.7564102564102564, 0.7371794871794872, 0.7435897435897436, 0.7307692307692307]
# accuracies1 (150x150) = [0.7516025641025641, 0.7932692307692307, 0.7467948717948718, 0.7788461538461539, 0.7371794871794872, 0.7548076923076923, 0.7355769230769231, 0.75, 0.7307692307692307]