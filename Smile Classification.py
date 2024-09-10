#!/usr/bin/env python
# coding: utf-8

# # **SMILE CLASSIFICATION**

# # **OVERVIEW**
# 
# This notebook demonstrates classifying images of different faces into **three categories**. They are as follows:
# 
# 1.   **NOT smile** : The face doesn’t have a smile.
# 2.   **POSITIVE smile** : The face has a real smile.
# 3.   **NEGATIVE smile** : The face has a fake smile.
# 
# 
# 
# 

# In[ ]:


#Importing Libraries and modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from PIL import ImageOps
from pathlib import Path
import cv2
import gc
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.resnet import ResNet50
from keras.applications.vgg19 import VGG19
from sklearn.model_selection import train_test_split


# The images folder in the data is in .zip format. We use [zipfile](https://docs.python.org/3/library/zipfile.html) library to extract images.

# In[ ]:


import zipfile
with zipfile.ZipFile("/content/drive/MyDrive/Copy of happy_images.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/drive/MyDrive")


# We use [pathlib](https://docs.python.org/3/library/pathlib.html) module for file handling. This provides an object API for working with files and directories.

# In[ ]:


image_folder=Path("/content/drive/MyDrive/happy_images")


# Let's visualize some of the images

# In[ ]:


#Creating axes
fig,ax=plt.subplots(7,7,figsize=(20,20))
ax=ax.ravel()
i=0

#iterdir allows us to iterate the files in the directory
for entries in image_folder.iterdir():
  image=Image.open(str(entries))
  ax[i].imshow(image)
  if i==48:
    break
  i+=1


# In[ ]:


train=pd.read_csv("/content/drive/MyDrive/Copy of train.csv",header=None)
test=pd.read_csv("/content/drive/MyDrive/Copy of test.csv",header=None)


# In[ ]:


train[1].value_counts()


# In[ ]:


# Check for class imbalance
sns.countplot(x=train[1])


# From the above graph, it is evident that there is **class imbalance** in our data. To avoid this problem, let's perform **oversampling** of minority classes.

# # **BALANCING CLASSES**

# In[ ]:


def balance_classes(df,split_ratio):
    """This function oversamples the minority classes to balance the data"""

    # We are splitting the data before oversmapling to avoid data leakage problem.
    train_df=df.iloc[:int(len(df)*(1-split_ratio)),:]
    test_df=df.iloc[int(len(df)*(1-split_ratio)):,:]

    # Getting the number of samples to required to balance the classes 
    extra_pos_smile=sum(train_df["label"]=="NOT smile")-sum(train_df["label"]=="positive smile")
    extra_neg_smile=sum(train_df["label"]=="NOT smile")-sum(train_df["label"]=="negative smile")
    
    # Sampling the required number of rows from train dataframe
    pos_smile_samples=train_df[train_df["label"]=="positive smile"].sample(extra_pos_smile,replace=True,random_state=0)
    neg_smile_samples=train_df[train_df["label"]=="negative smile"].sample(extra_neg_smile,replace=True,random_state=0)
    
    # Concatenating the sampled data
    train_df=pd.concat([train_df,pos_smile_samples,neg_smile_samples],ignore_index=True)
    
    # Shuffling the dataframe
    train_df=train_df.sample(frac=1)
    
    # Resetting the indices of the train and test dataframes
    train_df.reset_index(inplace=True,drop=True)
    test_df.reset_index(inplace=True,drop=True)
    
    return train_df,test_df


# Now, Let's preprocess the images to improve the **quality** so that we can analyse it in a better way and to suppress **undesired distortions** and enhance some features which are necessary for our classification.

# # **IMAGE PROCESSING**

# In[ ]:


def preprocess_image(path, image_width, image_height):

    image=cv2.imread(path)

    # Converting the image from RGB band to BGR
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    # Resizing image to a fixed image_width and image_height
    image=cv2.resize(image,(image_width,image_height))
    
    # Applying clahe equalization to each channel to improve contrast and brightness of the image
    clahe=cv2.createCLAHE(clipLimit = 4,tileGridSize=(2,2))
    for channel in range(3):
      image[:, :, channel] = clahe.apply(image[:, :, channel])

    image = cv2.fastNlMeansDenoisingColored(image, None, 10,10,7,21)
    
    # Applying dilation to whiten teeth pixels and remove black tint for better feature extraction
    kernel = np.ones((3,3),np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)

    # Sharpen the image to remove blur and highlight the edges of key features
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image = cv2.filter2D(image, -1, sharpen_kernel)
    
    return image


# Finally, get the image arrays from the happy_images directory

# # **INPUT PREPROCESSING**

# In[ ]:


def input_preprocess(dataframe,image_width,image_height):
    """This function converts the images into arrays"""
    
    df=dataframe.copy()

    #Renaming the labels of dataframe
    df.columns=["id","label"]
    
    # Balancing classes
    train_df,test_df=balance_classes(df,split_ratio=0.15)
    
    # Converting train dataframe into arrays
    X_train=[];i=0;y_train=[]
    for filename in train_df["id"].values:
        path = "/content/drive/MyDrive/happy_images/"+filename+".jpg"
        image = preprocess_image(path, image_width, image_height)
        X_train.append(image)
        y_train.append(train_df["label"][i])

        i+=1

        # Check
        if i%100==0:
          print(i,"train images completed")
    
    # Converting test dataframe into arrays
    X_test=[];i=0;y_test=[]
    for filename in test_df["id"].values:
        path="/content/drive/MyDrive/happy_images/"+filename+".jpg"
        image = preprocess_image(path, image_width, image_height)
        X_test.append(image)
        y_test.append(test_df["label"][i])
        
        i+=1

        if i%100==0:
          print(i,"test images completed")
    
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    
    # One Hot encoding of labels
    y_train=pd.get_dummies(y_train).values
    y_test=pd.get_dummies(y_test).values  
    
    return X_train, X_test, y_train, y_test


# In[ ]:


X_train, X_test, y_train, y_test = input_preprocess(train, 150,150)


# In[ ]:


# Save the files
np.save("/content/drive/MyDrive/X_train.npy", X_train)
np.save("/content/drive/MyDrive/X_test.npy", X_test)
np.save("/content/drive/MyDrive/y_train.npy", y_train)
np.save("/content/drive/MyDrive/y_test.npy", y_test)


# In[ ]:


X_train = np.load("/content/drive/MyDrive/X_train.npy")
X_test = np.load("/content/drive/MyDrive/X_test.npy")
y_train = np.load("/content/drive/MyDrive/y_train.npy")
y_test = np.load("/content/drive/MyDrive/y_test.npy")


# In[ ]:


# Checking the shapes
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # **PREPROCESS TEST DATA**

# In[ ]:


def preprocess_test(dataframe, image_width, image_height):
    df=dataframe.copy()

    #Renaming the labels of dataframe
    df.columns=["id","label"]
    
    # Converting train dataframe into arrays
    test_X=[];i=0;test_y=[]
    for filename in df["id"].values:
        path = "/content/drive/MyDrive/happy_images/"+filename+".jpg"
        image = preprocess_image(path, image_width, image_height)
        test_X.append(image)
        test_y.append(df["label"][i])

        i+=1

        # Check 
        if i%100==0:
          print(i, "samples done")

    test_X = np.array(test_X)
    test_y = pd.get_dummies(test_y).values

    return test_X, test_y


# In[ ]:


test_X, test_y = preprocess_test(test, 150,150)
test_X.shape, test_y.shape


# In[ ]:


np.save("/content/drive/MyDrive/test_X.npy", test_X)
np.save("/content/drive/MyDrive/test_y.npy", test_y)


# In[ ]:


test_X = np.load("/content/drive/MyDrive/test_X.npy")
test_y = np.load("/content/drive/MyDrive/test_y.npy")


# # **MODEL** **DEVELOPMENT**

# In[ ]:


def build_model(pretrained_net, input_shape, classes):

    if pretrained_net=="Efficient_net":
      base_model = EfficientNetB7(include_top=False, weights="imagenet", input_shape=input_shape,classes=classes)

    elif pretrained_net=="vgg19":
      base_model = VGG19(include_top=False, weights="imagenet", input_shape=input_shape,classes=classes)

    elif pretrained_net == "resnet":
      base_model = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape,classes=classes)

    flatten_layer=Flatten()
    dense_layer1=Dense(128,activation="relu")
    dropout_layer=Dropout(0.2)
    dense_layer2=Dense(64,activation="relu")
    prediction_layer=Dense(3,activation="softmax")
    
    model = Sequential([
        base_model,
        flatten_layer,
        dense_layer1,
        dropout_layer,
        dense_layer2,
        prediction_layer])

    return model


# # **ResNet50**

# In[ ]:


from keras.applications.resnet import preprocess_input

adam=keras.optimizers.Adam(learning_rate = 0.0001)

resnet_model = build_model(pretrained_net = "resnet", input_shape=X_train[0].shape, classes=y_train.shape[1])

resnet_model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

X_train_resnet = preprocess_input(X_train)
X_test_resnet = preprocess_input(X_test)

resnet_history= resnet_model.fit(X_train_resnet, y_train, batch_size=32, 
                                 epochs=10, validation_data=(X_test_resnet,y_test))


# In[ ]:


# Save the model
resnet_model.save("/content/drive/MyDrive/resnet.h5")


# Plot the accuracy and training plots

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.set()

loss_train = resnet_history.history['loss']
loss_val = resnet_history.history['val_loss']
epochs = range(1,11)
ax[0].plot(epochs, loss_train, 'r', label='Training loss')
ax[0].plot(epochs, loss_val, 'b', label='validation loss')
ax[0].set_title('Resnet model Training and validation loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

loss_train = resnet_history.history['accuracy']
loss_val = resnet_history.history['val_accuracy']
epochs = range(1,11)
ax[1].plot(epochs, loss_train, 'r', label='Training accuracy')
ax[1].plot(epochs, loss_val, 'b', label='validation accuracy')
ax[1].set_title('Resnet model Training and Validation accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()


# In[ ]:


# Test the ResNet model
resnet_model.evaluate(test_X, test_y)


# # **VGG 19**

# In[ ]:


from keras.applications.vgg19 import preprocess_input

adam=keras.optimizers.Adam(learning_rate = 0.0001)

vgg_model = build_model(pretrained_net = "vgg19", input_shape=X_train[0].shape, classes=y_train.shape[1])

vgg_model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])


X_train_vgg = preprocess_input(X_train)
X_test_vgg = preprocess_input(X_test)

vgg_history= vgg_model.fit(X_train_vgg, y_train, batch_size=32, 
                                 epochs=10, validation_data=(X_test_vgg,y_test))


# In[ ]:


vgg_model.save("/content/drive/MyDrive/vgg19.h5")


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.set()

loss_train = vgg_history.history['loss']
loss_val = vgg_history.history['val_loss']
epochs = range(1,11)
ax[0].plot(epochs, loss_train, 'r', label='Training loss')
ax[0].plot(epochs, loss_val, 'b', label='validation loss')
ax[0].set_title('VGG model Training and validation loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

loss_train = vgg_history.history['accuracy']
loss_val = vgg_history.history['val_accuracy']
epochs = range(1,11)
ax[1].plot(epochs, loss_train, 'r', label='Training accuracy')
ax[1].plot(epochs, loss_val, 'b', label='validation accuracy')
ax[1].set_title('VGG model Training and Validation accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()


# In[ ]:


vgg_model.evaluate(test_X, test_y)


# # **EfficientNet B7**

# In[ ]:


from keras.applications.efficientnet import preprocess_input

adam=keras.optimizers.Adam(learning_rate = 0.0001)

efficientnet_model = build_model(pretrained_net = "Efficient_net", input_shape=X_train[0].shape, classes=y_train.shape[1])

efficientnet_model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])


X_train_efficientnet = preprocess_input(X_train)
X_test_efficientnet = preprocess_input(X_test)

efficientnet_history= efficientnet_model.fit(X_train_efficientnet, y_train, batch_size=32, 
                                 epochs=10, validation_data=(X_test_efficientnet,y_test))


# In[ ]:


efficientnet_model.save("/content/drive/MyDrive/Efficient_Net.h5")


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.set()

loss_train = efficientnet_history.history['loss']
loss_val = efficientnet_history.history['val_loss']
epochs = range(1,11)
ax[0].plot(epochs, loss_train, 'r', label='Training loss')
ax[0].plot(epochs, loss_val, 'b', label='validation loss')
ax[0].set_title('Efficient-Net model Training and validation loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()

loss_train = efficientnet_history.history['accuracy']
loss_val = efficientnet_history.history['val_accuracy']
epochs = range(1,11)
ax[1].plot(epochs, loss_train, 'r', label='Training accuracy')
ax[1].plot(epochs, loss_val, 'b', label='validation accuracy')
ax[1].set_title('Efficient-Net model Training and Validation accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()


# In[ ]:


efficientnet_model.evaluate(test_X, test_y)

