import numpy as np
from keras import layers
from keras.layers import Input, Add,Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.layers.merge import concatenate,add
from cnn_blocks import VGG_block 

INPUT_SHAPE=(224,224,1)
n_classes=10
input=Input(INPUT_SHAPE)

block1=VGG_block(input,filters=64,no_of_conv_layers=2)
block2=VGG_block(block1,filters=128,no_of_conv_layers=2)
block3=VGG_block(block2,filters=256,no_of_conv_layers=3)
block4=VGG_block(block3,filters=512,no_of_conv_layers=3)
block5=VGG_block(block4,filters=512,no_of_conv_layers=3)
flatten_output=Flatten()(block5)
FC1=Dense(4096,activation='relu')(flatten_output)
FC2=Dense(4096,activation='relu')(FC1)
FC3=Dense(4096,activation='relu')(FC2)
output=Dense(n_classes,activation='softmax')(FC3)
model_vgg16=Model(inputs=input,outputs=output)
model_vgg16.summary()

model_vgg16.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

"""
For training the model:
(xtrain,ytrain),(xtest,ytest)=load_data()

#building the input vector 
Y_train = np_utils.to_categorical(ytrain, output_images)
Y_test = np_utils.to_categorical(ytest, output_images)
X_train=X_train.reshape(-1,224,224,1)
X_test=X_test.reshape(-1,224,224,1)
# normalizing the data to help with the training
X_train =xtrain/255
X_test = xtest/255
# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
print("Train matrix shape", Y_train.shape)
print("Test matrix shape", Y_test.shape)

model_vgg16.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Hyperparameters
EPOCHS=100
BATCH_SIZE=128
classifier=model_vgg16.fit(X_train,Y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,validation_data=(X_test,Y_test))
"""
model_vgg16.save("your_file_path/model_vgg16.h5",save_format='h5')
