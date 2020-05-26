import os
origin = '/Users/jaoming/Documents/Active Projects/Image Classification'
os.chdir(origin)

import numpy as np                               # images work better in arrays rather than dataframes
import cv2                                       # for importing the image
from keras.applications import MobileNetV2       # for applying the CNN model
from keras import models, layers
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
import random                                    # for randomizing the dataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# DATA PREPROCESSING
"""
Creating training and test sets
"""
os.chdir('catdog dataset')
dog_data, cat_data = [], []
for image in os.listdir():
       animal = image.split('.')[0]
       if animal == 'dog':
              dog_data.append(image)
       elif animal == 'cat':
              cat_data.append(image)

# splitting into training and test sets
random.shuffle(dog_data)
random.shuffle(cat_data)

dog_data = dog_data[:1000]
cat_data = cat_data[:1000]

train_data = dog_data[:750] + cat_data[:750]
test_data = dog_data[750:] + cat_data[750:]

del dog_data         # just to clean up the memory
del cat_data         # just to clean up the memory


# DATA TRANSFORMATION
random.shuffle(train_data)
random.shuffle(test_data)

train_X, train_y = [], []
test_X, test_y = [], []

# target image shape
height = 200         # number of rows
width = 200          # number of columns

# working on the training set
for image in train_data:
       animal = image.split('.')[0]
       train_X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR),
                                   (height, width),
                                   interpolation = cv2.INTER_CUBIC))
       if animal == 'cat':
              train_y.append(1)
       elif animal == 'dog':
              train_y.append(0)

# working on the test set
for image in test_data:
       animal = image.split('.')[0]
       test_X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR),
                                   (height, width),
                                   interpolation = cv2.INTER_CUBIC))
       if animal == 'cat':
              test_y.append(1)
       elif animal == 'dog':
              test_y.append(0)

del train_data       # just to clean up memory
del test_data        # just to clean up memory

train_X, train_y = np.array(train_X), np.array(train_y)
test_X, test_y = np.array(test_X), np.array(test_y)

# IMPLEMENTATION OF THE PRE MADE MODEL
base_model = MobileNetV2(include_top = False, input_shape = (height, width, 3)) # include_top means the output layer

## establishing the output layers
out = base_model.output
out = layers.Flatten()(out)
out = layers.Dense(128, 
                     activation = 'relu',
                     kernel_initializer = 'he_uniform')(out)
out = layers.Dense(1, activation = 'sigmoid')(out)
## establishing the input layers
inp = base_model.input

final_model = models.Model(inp, out)

rmsprop = RMSprop(learning_rate = 0.001, rho = 0.9)

final_model.compile(optimizer = rmsprop,
              loss = 'binary_crossentropy',
              metrics = ['acc'])
              
train_datagen = ImageDataGenerator(rotation_range = 40, # data filter. rotates/flips/manipulates image that's passed through here
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_generator = train_datagen.flow(train_X, train_y, batch_size = 32)

history = final_model.fit_generator(train_generator,
                            steps_per_epoch = len(train_y)/32,                      # so the model trains per batch
                            epochs = 1,
                            callbacks = [EarlyStopping(monitor = 'val_loss', patience = 0, min_delta = 0.0001)])

y_pred = final_model.predict(test_X)
y_pred = np.where(y_pred > 0.5, 1, 0)

print(accuracy_score(test_y, y_pred))

def summarize_diagnostics(history):
	# plot loss and accuracy
       plt.title('Cross Entropy Loss and Classification Accuracy')
       plt.plot(history.history['loss'], color = 'blue', label = 'loss')
       plt.plot(history.history['acc'], color = 'red', label = 'accuracy')
       plt.legend()
summarize_diagnostics(history)