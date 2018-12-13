#importing packages and libraries required
from keras.models import Sequential #import model
from keras.layers.core import Dense,Dropout, Activation, Flatten #import Layers
from keras.layers.convolutional import Convolution2D, MaxPooling2D #Import Conv layers
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, adam
from itertools import repeat
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
#Sklearn
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
#%%
#Preprocessing
path_Train = 'F:\ZewailCity\ZU\Year 5\Semester 2\Computer Vision\Projects\Final Project\GTSRB\Final_Training\Images'
path_Train_Resized = 'F:\ZewailCity\ZU\Year 5\Semester 2\Computer Vision\Projects\Final Project\GTSRB\Final_Training\Images_Resized'
listing_train = os.listdir(path_Train)
j=99
k=0
label = []
list_train = []
for file in listing_train:
    list_train= os.listdir(path_Train + '\\'+ file)
    j-=1
    for f in list_train:
        im = Image.open(path_Train+'\\'+ file +'\\'+ f)
        img = im.resize((32,32))
        gray = img.convert('L')
        gray.save(path_Train_Resized+'\\'+ str(j) + f,"JPEG")        

#%%
imlist = os.listdir(path_Train_Resized)

#number of images
imnbr = len(imlist)

#Create Matrix for flattenig all images
immatrix = np.array([np.array(Image.open(path_Train_Resized + '\\' + im2)).flatten()
            for im2 in imlist],'f')

#Labeling Training set
label = np.ones((imnbr,),dtype = int)
k=0
label[k]=0
m=imlist[0][0]
n=imlist[0][1]
for h in imlist:
    if k < len(imlist)-1:
        k+=1
    if h[0]==m and h[1]==n:
        label[k]=label[k-1]
    else:
        label[k]=label[k-1]+1
    m=h[0]
    n=h[1]
#Testing
img = immatrix[211].reshape(32,32)
plt.imshow(img,cmap='gray')
#Shuffling
data,Label = shuffle(immatrix,label, random_state = 2)
train_data = [data,Label]
#%%
#Model Parameters
#batch size to train
batch_size = 30
#number of output classes
nb_classes = 43
#number of trainings on all data set
nb_epochs = 15
#input image dimensions
img_rows , img_cols = 32,32
#number of conv filters to use
nb_filters = 32
#size of pooling area for max pooling 
nb_pool = 2
#conv kernel size
nb_conv = 3
#%%
(X,y) = (train_data[0], train_data[1])

#Validation by splitting train dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

X_train = X_train.reshape(X_train.shape[0],1, img_rows,img_cols)
X_test = X_test.reshape(X_test.shape[0],1, img_rows,img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Normalization
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#%%
# CNN model
model = Sequential ()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode ='valid',
                        input_shape = (1,img_rows, img_cols)))
#activation function rectified linear units
convout1 = Activation ('relu')
model.add(convout1)

model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation ('relu')
model.add(convout2)

model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout3 = Activation ('relu')
model.add(convout3)

model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout4 = Activation ('relu')
model.add(convout4)

model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout4 = Activation ('relu')
model.add(convout4)

#MaxPooling layer
model.add(MaxPooling2D(pool_size = (nb_pool,nb_pool)))
model.add(Dropout(0.25))

#Fully - Connected Layer
model.add(Flatten())
model.add(Dense(43))
model.add(Activation ('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation ('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics=['accuracy'])
#%%
#Training the model and getting validation accuracy after each epoch
model.fit(X_train, Y_train, batch_size=batch_size , nb_epoch=nb_epochs,
              verbose = 1, validation_data = (X_test, Y_test))

#%%
#Test dataset
#Preprocessing
path_Test = 'F:\ZewailCity\ZU\Year 5\Semester 2\Computer Vision\Projects\Final Project\GTSRB\Final_Test\Images'
path_Test_Resized = 'F:\\ZewailCity\\ZU\\Year 5\\Semester 2\\Computer Vision\\Projects\\Final Project\\GTSRB\\Final_Test\\Images_Resized'
listing_test = os.listdir(path_Test)
list_test = []
for f in listing_test:
        im = Image.open(path_Test+'\\'+ f)
        img = im.resize((32,32))
        gray = img.convert('L')
        gray.save(path_Test_Resized+'\\'+ f,"JPEG")
imlistt2 = os.listdir(path_Test_Resized)
imnbr2 = len(imlistt2)
#Create Matrix for flattenig all images
immatrix2 = np.array([np.array(Image.open(path_Test_Resized + '\\' + im3)).flatten()
            for im3 in imlistt2],'f')
#%%
#Reading Test Data Labels
Test_Labels_Path = 'F:\ZewailCity\ZU\Year 5\Semester 2\Computer Vision\Projects\Final Project\\test_label.txt'
test_labels = np.ones((imnbr2,),dtype = int)
fileHandle = open ( Test_Labels_Path, 'r' )
test_labels = fileHandle.readlines()
fileHandle.close()
test_labels = map(int, test_labels)
test_labels = np.array(test_labels)
test_labels = 42 - test_labels
#%%
#Preparing test data
data1,Label1 = immatrix2,test_labels
test_data = [data1,Label1]

(X_test2,Y_test2) = (test_data[0], test_data[1])

X_test2 = X_test2.reshape(X_test2.shape[0],1, img_rows,img_cols)

X_test2 = X_test2.astype('float32')

X_test2 /= 255

Y_test2 = np_utils.to_categorical(Y_test2, nb_classes)

#%%
#Evaluating model with test data
score =  model.evaluate(X_test2,Y_test2,verbose = 0)
print('Test accuracy : ',score[1])
















