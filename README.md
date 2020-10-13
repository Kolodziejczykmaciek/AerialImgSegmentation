# AerialImgSegmentation
# Table of contents
* [General info](#general-info)
* [Project structure](#structure)
* [Requirements](#requirements)
* [Preparing the dataset](#preparing-the-dataset)
* [Training the model](#training-the-model)
* [Segmentation](#segmentation)
* [Setup](#setup)

## General info
![GitHub Logo](/githubimg/seggg.png)

The project includes implementations of the unet model for semantic segmentation.

It is necessary to download the dataset together with the labels available for free https://project.inria.fr/aerialimagelabeling/.
Just fill in the form and you can download the dataset. The compressed dataset weighs around 20 GB! First dowload all five compressed file to your's computer folder,
then extract the first one and the rest 4 should extract automatically. 

## Project structure

## Requirements
Project is created with:
* dataset + labels https://project.inria.fr/aerialimagelabeling/
* python 3.7
* tensorflow==2.0
* numpy
* opencv-python
* matplotlib

## Preparing the dataset
The dataset is being parsed with aerialDatasetParser.py script.
```python
    def aerialDataTrain(num_train_img):
```
This method returns given number of training images and corresponding masks as a nd.array.
Images in data set are 5000x5000 pixels this method cuts every image and its binary mask into 512x512
training images.
One 5000x5000 pixel image is cut into 100 images.
For now the method is hardcoded to evaluate training 5000x5000 pixel images into 512x512 but
it can by transormed to return images in different resolution. As far as i konw the resolution must be always
divisible by 32.

```python
  def aerialValidationData(num_val_img):
```
This method returns given number of 100xvalidation images as a nd.array.
Images in data set are 5000x5000 pixels this method cuts every image into 512x512 validation images.
One 5000x5000 pixel image is cut into 100 images.
For now the method is hardcoded to evaluate validation 5000x5000 pixel images into 512x512 but
it can by transormed to return images in different resolution. As far as i konw the resolution must be always
divisible by 32.

## Training the model
```python
X_train, Y_train = aerialDataTrain(1000) #number of images you want to train the model here 1000
```
Actuall training starts here:
```python
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=100, callbacks=callbacks)
```
## Segmentation
The project contains one trained model in folder "Trained_models".
To run the already trained model use segmentation.py script.
The script first gets the validation data.
```python
X_val = aerialValidationData(1000) #again in brackets specify number of images you want to validate on
```
Second load the model.
```python
model = tf.keras.models.load_model(/path)
```
Third evaluate on loaded model.
```python
model.predict(X_val, verbose=1)
```
In script you will find also code which displays the segmentated images.
## Setup
