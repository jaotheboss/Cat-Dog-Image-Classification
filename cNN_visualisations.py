"""
Bottom line is that all the visualisations for a convolutional neural network
aims to find out what each filter (be it in the middle of the model or the output layer)
is focusing on. 

The visualisation of the mid-layers take on the method of edge detection
The saliency mapping takes a look at the ranks of each individual pixel
The Grad-CAM takes into account the gradients of each neuron at the output layer

Grad-CAM is by far the most mathematically sound in my perspective as it's relevant and simple to understand
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

# importing the machine learning model
os.chdir('models')
model = models.load_model('catdog_classifier_v1.h5', compile = False)
model.compile(optimizer = RMSprop(learning_rate = 0.001, rho = 0.85),
              loss = 'binary_crossentropy',
              metrics = ['acc'])
os.chdir(origin)

# import some image
os.chdir('catdog dataset')
image = cv2.imread('cat.0.jpg')
image = cv2.resize(cv2.imread('cat.0.jpg', cv2.IMREAD_COLOR),  # IMREAD_COLOR ensures 3 color channel input
                            (200, 200),                        # image target size
                            interpolation = cv2.INTER_CUBIC)   # the kind of interpolation done by resizing the image
os.chdir(origin)

# plotting out the image 
plt.imshow(image[:,:,0])

# Post-Prediction Visualisation 1 - to visualise the layers
def display_activation(activations, col_size, row_size, act_index):
       """
       ! Post-Prediction Visualisation 

       Function:     To show the patterns learnt by the layers of the c-NN

       Input:        activatons    - the model
                     col_size      - the number of columns to show the patterns
                     row_size      - the number of rows to show the patterns
                     act_index     - which layer to visualise
       """
       activation = activations[act_index]
       activation_index = 0
       fig, ax = plt.subplots(row_size, col_size, figsize = (row_size * 2.5, col_size * 1.5))
       for row in range(0, row_size):
              for col in range(0, col_size):
                     ax[row][col].imshow(activation[0, :, :, activation_index], cmap = 'gray')
                     activation_index += 1

layer_outputs = [layer.output for layer in model.layers]                                   # recreating the model
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)             # because we have to use the Model class
actvns = activation_model.predict(image.reshape(1, 200, 200, 3))                           # predicting one pic

for i in range(len(actvns)):
       if len(actvns[i].shape) > 2:
              print('Layer', str(i + 1), 'has\t', actvns[i].shape[-1], '\tFeature Maps')
       else:
              print('Layer', str(i + 1), 'has\t', actvns[i].shape[-1], '\tNodes')

display_activation(actvns, 8, 4, 0)                                                        # rows, columns, layer

# Post-Prediction Visualisation 2 - to visualise the output layer
## Saliency Maps = ranks the pixels based on how influential they are in the outcome
## ie. the parts that help determine what class the image is in
## https://arxiv.org/pdf/1312.6034.pdf

from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'dense_2')

# Swap softmax/sigmoid with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

# number of classes
classes = [0] # technically a sigmoid only has 1 class
## just refer to the number of output nodes 

# visualise
for c in classes:
       visualisation = visualize_activation(model, layer_idx = layer_index, filter_indices = c, input_range = (0, 1)) # input_range apparently sharpens the image
       plt.imshow(visualisation[..., 0])
       plt.title(f"Class = {c}")
       plt.show()

# Post-Prediction Visualisation 3 - to visualise the output layer
## Grad-CAM = stands for Gradient-weighted Class Activation Mapping
## it uses the gradients of the target concept that flows into the output layer to produce a 
## coarse localization map highlighting important regions in the image for predicting the concept
## https://arxiv.org/abs/1610.02391 helps with the explanation
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
import matplotlib.cm as cm

# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'dense_2')

# Swap softmax/sigmoid with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

# Visualize
no_of_pics = 1
for n in range(no_of_pics):
       # Get input
       input_image = image
       input_class = 0                                  # the class of the image

       # Matplotlib preparations
       fig, axes = plt.subplots(1, 3)

       # Generate visualization
       visualization = visualize_cam(model, layer_index, filter_indices = 0, seed_input = input_image)
       axes[0].imshow(input_image[..., 0], cmap = 'gray') 
       axes[0].set_title('Input')

       axes[1].imshow(visualization)
       axes[1].set_title('Grad-CAM')

       heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
       original = np.uint8(cm.gray(input_image[..., 0])[..., :3] * 255)
       axes[2].imshow(overlay(heatmap, original))
       axes[2].set_title('Overlay')

       fig.suptitle(f"Class = Cat")
       plt.show()