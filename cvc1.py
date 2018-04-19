
# coding: utf-8

# In[1]:


print("starting")
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2                
import numpy as np
import keras
from keras.preprocessing import image                  
from tqdm import tqdm
print("done import")


# In[2]:


def load_dataset(path):
    data = load_files(path)
    # print(data)
    face_files = np.array(data['filenames'])
    face_targets = np_utils.to_categorical(np.array(data['target']), 15)
    return face_files, face_targets

w,h = 64,64

import random
random.seed(9)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    try:
        img = image.load_img(img_path, target_size=(w,h))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    except IOError:
        pass
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# pre-process the data for Keras




# In[3]:


print("starting load data")
# load train, test, and validation datasets
train_files, train_targets = load_dataset('train')
test_files, test_targets = load_dataset('validation')
print("done load data")
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# In[4]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import keras
model = Sequential()

### TODO: Define your architecture
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),input_shape=(w,h,3),activation='tanh'))
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),activation='tanh'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='tanh'))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# model.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))

# model.add(GlobalAveragePooling2D())

model.add(Flatten())
model.add(Dense(256,activation='tanh'))

model.add(Dropout(0.7))
model.add(Dense(15,activation='softmax'))


model.summary()

rmsprop = keras.optimizers.RMSprop(lr=0.00001,rho=0.5)
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])


# In[5]:


from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs =500

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='cvc123.hdf5',
                               verbose=1, save_best_only=True)



# In[ ]:


model.fit(train_tensors, train_targets,validation_data=(test_tensors,test_targets),epochs=epochs, batch_size=64, callbacks=[checkpointer], verbose=1)
model.save('model.hdf5')
# #
#
model=keras.models.load_model('/mnt/attic/Projects/cnn_training/saved_models/cvc123.hdf5')
#


# In[ ]:


score=model.evaluate(test_tensors,test_targets)
print('test loss:',score[0])
print('test accuracy:',score[1])

