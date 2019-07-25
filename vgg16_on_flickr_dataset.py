
# coding: utf-8

# In[20]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import layers
from keras import models
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras import optimizers
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

base_path = '/Users/Vineet/Documents/Courses/ELEN 297/Flickr/images/'


# In[21]:


# Set Paths
train_dir = base_path+'/train/'
validation_dir = base_path+'/validation/'
test_dir = base_path+'/test/'


# In[22]:


#Build Model
model = VGG16(include_top=True,weights=None,input_shape=(150, 150, 3), classes=10)
model.summary()


# In[23]:


#Preprocess data
train_datagen = ImageDataGenerator(rescale=1./255)             
test_datagen = ImageDataGenerator(rescale=1./255)              

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150), 
        batch_size=20, 
        class_mode='categorical')                                   

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

