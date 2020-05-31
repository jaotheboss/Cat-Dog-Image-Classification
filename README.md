# Cat-Dog-Image-Classification

![Cat](https://github.com/jaotheboss/Cat-Dog-Image-Classification/blob/master/cat-gradcam.jpeg)

Using convolutional neural networks to distinguish between images of cats and dogs. 

## Objective:
Simple introduction and practice into the world of deep learning with images. In this project, i brush up my skills on working with images and image classification models

## Left To-do:
- [x] Improve Accuracy to above 0.7
- [ ] Mid-Model Visualisations 
- [x] Post-Prediction Visualisations

#### Visualisations Packages for c-NN:
1. https://github.com/raghakot/keras-vis (be sure to install the right version. some methods would download the deprecated version)

#### Post-Prediction Visualisations to try:
1. https://neilnie.com/2018/04/13/visualizing-the-steering-model-with-attention-maps/ (dependent on keras-vis)

2. https://www.machinecurve.com/index.php/2019/11/25/visualizing-keras-cnn-attention-saliency-maps/ (dependent on keras-vis)

## Methodology:
In this project, I use a 3 block VGG architecture convnet model to classify the images. VGG has been a classic image classification architecture for quite some time now and i intend to play around with the other parameters to yield better results from smaller datasets.

## Notes:
1. There are various ways to import an image into data. For this project, i used OpenCV to import the image. This is because using Keras's in-built conversion function can cause problems with the model according to some research articles online.

2. It is a practice to have the pixel value vary between 0-1 rather than 0-255.

3. Without an ImageDataGenerator, it is worth noting that the CNN model takes in numpy arrays only. No lists allowed.

4. An ImageDataGenerator basically acts as a filter for the flow of images. So the logic for this would be to
  * Create the filter for either test or train with the relevant parameters. So it could be: train_datafilter = ImageDataGenerator(scale = 1/255, zoom = 0.1, shift_left = 0.1, ...)
  * Afterwhich, you pass the actual data through this filter to create an instance that generates the image that's been passed through the filter. So it could be: train_imagegenerator = train_datafilter(train_X, train_y, batch_size = 32)
  * Instead of using model.fit(...), we use model.fit_generator(...)
  
This is the general idea for using a CNN model. There are other forms of architecture to make use of and other methods to implement to improve the current model like: Transfer Learning, Dropout, BatchNormalization. In fact, ImageDataGenerator is one of those tools that help improve the model. 
