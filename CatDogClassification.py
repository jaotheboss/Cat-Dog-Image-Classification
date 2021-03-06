"""
we'll be referencing:
https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9

https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

# for working with data directly from directory (will waste less RAM)
https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
"""

import os
origin = '/Users/jaoming/Documents/Active Projects/Image Classification'
os.chdir(origin)

import numpy as np                               # images work better in arrays rather than dataframes
import cv2                                       # for importing the image
from keras import models                         # for building the deep learning model
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adadelta
import random                                    # for randomizing the dataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

os.chdir('catdog dataset')

"""
importing an image using keras. not recommended due to a bug when resizing

image = keras.preprocessing.image.load_img('cat.0.jpg',
                                          grayscale = False,
                                          target_size = (150, 150))
input_arr = keras.preprocessing.image.img_to_array(image) # x, y, z where z represents the color scale
input_arr = np.array([input_arr]) # converts a single image to a batch
"""
# testing things out
image = cv2.imread('cat.0.jpg')
image.shape # shows the shape of the image (height, width, colors)
image[100, 100] # accesses an individual pixel (Blue, Green, Red)
image.size # shows the total number of pixels in the image

image = cv2.resize(cv2.imread('cat.0.jpg', cv2.IMREAD_COLOR),  # IMREAD_COLOR ensures 3 color channel input
                            (150, 150),                        # image target size
                            interpolation = cv2.INTER_CUBIC)   # the kind of interpolation done by resizing the image

# DATA PREPROCESSING
"""
Creating training and test sets
"""
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

# DATA MINING
train_X, train_y = np.array(train_X), np.array(train_y)
test_X, test_y = np.array(test_X), np.array(test_y)
os.chdir(origin)

# Building the Deep Learning model
## Three Block VGG Architecture model
model = models.Sequential()        # models.load_model('path/to/location')
model.add(layers.Conv2D(filters = 32,
       kernel_size = (3, 3),
       activation = 'relu',
       kernel_initializer = 'he_uniform', # or random_normal
       input_shape = (height, width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(filters = 64,
       kernel_size = (3, 3),
       activation = 'relu',
       kernel_initializer = 'he_uniform'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(filters = 128,
       kernel_size = (3, 3),
       activation = 'relu',
       kernel_initializer = 'he_uniform'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()

rmsprop = RMSprop(learning_rate = 0.001, rho = 0.85)                   # 0.64        0.694
adadelta = Adadelta(learning_rate = 0.8, rho = 0.95)                  # 0.62        0.598

lr_reduction = ReduceLROnPlateau(monitor = 'val_acc',
       patience = 2, 
       verbose = 1, 
       factor = 0.5, 
       min_lr = 0.00001)
early_stopping = EarlyStopping(monitor = 'val_loss', 
       patience = 0, 
       min_delta = 0.00001)

model.compile(optimizer = rmsprop,
              loss = 'binary_crossentropy',
              metrics = ['acc'])

train_datagen = ImageDataGenerator(rescale = 1/255,
                                   rotation_range = 30, # data filter. rotates/flips/manipulates image that's passed through here
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   shear_range = 0.1,
                                   zoom_range = 0.1,
                                   horizontal_flip = True)
train_generator = train_datagen.flow(train_X, train_y, batch_size = 24)

history = model.fit_generator(train_generator,
                            steps_per_epoch = len(train_y)/24,                      # so the model trains per batch
                            epochs = 5,
                            callbacks = [early_stopping, lr_reduction])

y_pred = model.predict(test_X)
y_pred = np.where(y_pred > 0.5, 1, 0)

print(accuracy_score(test_y, y_pred))

# PLOTTING RESULTS FROM THE NEURAL NETWORK
def summarize_diagnostics(history):
	# plot loss and accuracy
       plt.title('Cross Entropy Loss and Classification Accuracy')
       plt.plot(history.history['loss'], color = 'blue', label = 'loss')
       plt.plot(history.history['acc'], color = 'red', label = 'accuracy')
       plt.legend()
summarize_diagnostics(history)

## visualizing the mid training process
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

layer_outputs = [layer.output for layer in model.layers]                            # recreating the model
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)      # because we have to use the Model class
activations = activation_model.predict(train_X[1].reshape(1, 200, 200, 3))          # predicting one pic

plt.imshow(train_X[1][:,:,0]) # plot the image at layer 0 (doesn't matter if its grayscale or not)
display_activation(activations, 8, 4, 2)        # rows, columns, layer
# check activations[i].shape[3] to see how many feature maps there are for layer i
for i in range(len(activations)):
    print(activations[i].shape)

"""
Conventional training route
Note: This requires your pixel value to be between 0 and 1 already, instead of 0 and 255.

model.fit(x = train_X, y = train_y, 
       batch_size = 32, epochs = 5,
       validation_split = 0.1, 
       callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2, min_delta = 0.0001)]) 
"""

"""
ImageDataGenerator() will help rescale and apply inconsistencies in the image for better training

train_datagen = ImageDataGenerator(rescale = 1/255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_generator = train_datagen.flow(train_X, train_y, batch_size = 32)
# train_datagen.flow_from_directory('directory', class_mode = 'binary', batch_size = 32, target_size = (height, width))
model.fit_generator(train_generator,
                     steps_per_epoch = len(train_y)/32, # so the model trains per batch
                     epochs = 24)
# the train_generator only generates the images when its about to be fed to the model
# this method would work well if there are very few datapoints in the dataset

# using the imagedatagenerator for test set and evaluation
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(test_X, test_y, batch_size = 32)
int(model.evaluate_generator(test_generator)[1]*1000)/1000
"""

os.chdir('models')
model.save('catdog_classifier_v1')
model.save_weights('catdog_classifier_v1') # about half the size
os.chdir(origin)
