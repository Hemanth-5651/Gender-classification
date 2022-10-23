# gender clasification using male and female datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
%matplotlib inline
#issues are cleared
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array

inception_weights_path="https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"

!git clone https://github.com/laxmimerit/male-female-face-dataset.git

epochs =50
lr=1e-3# learning rate is 10^-3
batch_size=128
data=[]
labels=[]

size=224 
train_datagen=ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.4,
                             height_shift_range=0.4,
                             zoom_range=0.3,
                             rotation_range=20,
                             rescale=1/255 )
                             
test_gen=ImageDataGenerator(rescale=1/255)
target_size=(size,size)

target_size
train_generator=train_datagen.flow_from_directory(
    directory='/content/male-female-face-dataset/Training',
    target_size=target_size,

    batch_size=batch_size,
    class_mode='binary'
)

validation_generator=test_gen.flow_from_directory(
    directory='/content/male-female-face-dataset/Validation',
    target_size=target_size,

    batch_size=batch_size,
    class_mode='binary'
)

train_generator.class_indices

train_generator.classes

train_generator.class_mode

x,y=train_generator.next()

model=Sequential()
model.add(InceptionV3(include_top=False,pooling='avg',weights='imagenet'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))
model.layers[0].trainable=False

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )
model.fit(train_generator,steps_per_epoch=len(train_generator.filenames)//batch_size,epochs=2,validation_data=validation_generator,validation_steps=len(validation_generator.filenames)//batch_size)

img_path='/content/photo.jpg'
def main(path):
  img=load_img(path,target_size=(size,size,3))
  plt.imshow(img)
  img=img_to_array(img)
  img=img/255
  img=img.reshape(1,size,size,3)
  return prob(img)
  
print(main(img_path))
def prob(img):
  prob=model.predict(img)[0]
  if prob<0.5:
    return 'female','prob=',1-prob[0]
  else:
    return 'male','prob=',prob[0]


import imutils
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
  
  image_file = take_photo()
  #image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
image = cv2.imread(image_file)


# resize it to have a maximum width of 400 pixels
image = imutils.resize(image, width=400)

(h, w) = image.shape[:2]
print(w,h)
cv2_imshow(image)

import pickle
  
# Save the trained model as a pickle string.
saved_model = pickle.dumps(knn)
