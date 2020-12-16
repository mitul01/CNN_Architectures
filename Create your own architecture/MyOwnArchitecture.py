import numpy as np
from keras import layers
from keras.layers import Input, Add,Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D,Dropout
from keras.models import Model
from keras.layers.merge import concatenate,add
from cnn_blocks import inception_residual_block_A,inception_residual_block_B,inception_residual_block_C
from cnn_blocks import inception_block_reduction_A,inception_block_reduction_B
from cnn_blocks import residual_block_v2

n_classes=10

def stem(input):
	""" 
	A stem is the starting point of any CNN arch. It's output is followed by various inception modules.
	
	"""  
	l1=Conv2D(32,(3,3),strides=(2,2),activation='relu',padding='same')(input)
	l2=Conv2D(32,(3,3),strides=(2,2),activation='relu',padding='same')(l1)
	l3=Conv2D(64,(3,3),activation='relu',padding='same')(l2)
	l4_1=Conv2D(96,(3,3),strides=(2,2),activation='relu',padding='same')(l3)
	l4_2=Conv2D(96,(3,3),strides=(2,2),activation='relu',padding='same')(l3)
	l5=concatenate([l4_1,l4_2])

	l6_1=Conv2D(64,(1,1),activation='relu',padding='same')(l5)
	l6_2=Conv2D(96,(3,3),strides=(2,2),activation='relu',padding='same')(l6_1)

	l7_1=Conv2D(64,(1,1),activation='relu',padding='same')(l5)
	l7_2=Conv2D(64,(7,1),activation='relu',padding='same')(l7_1)
	l7_3=Conv2D(64,(1,7),activation='relu',padding='same')(l7_2)
	l7_4=Conv2D(96,(3,3),strides=(2,2),activation='relu',padding='same')(l7_3)

	l8=concatenate([l6_2,l7_4])

	l9_1=Conv2D(192,(3,3),strides=(2,2),activation='relu',padding='same')(l8)
	l9_2=MaxPooling2D((3,3),strides=(2,2),padding='same')(l8)

	output=concatenate([l9_1,l9_2])

	return output

input=Input(shape=(264,264,3))
stem_output=stem(input)
# Branch 1
branch1_1=inception_residual_block_A(stem_output)
branch1_2=inception_residual_block_A(branch1_1)
branch1_3=inception_residual_block_A(branch1_2)
pool1=AveragePooling2D()(branch1_3)
fc1=Dense(1024,activation='relu')(pool1)
output1=Dense(n_classes,activation='softmax')(fc1)
# Branch 2
branch2_1=inception_residual_block_B(stem_output)
branch2_2=inception_residual_block_B(branch2_1)
branch2_3=inception_residual_block_B(branch2_2)
branch2_4=inception_residual_block_B(branch2_3)
branch2_5=inception_residual_block_B(branch2_4)
# Branch 3
branch3_1=inception_residual_block_C(stem_output)
branch3_2=inception_residual_block_C(branch3_1)
branch3_3=inception_residual_block_C(branch3_2)

l1=concatenate([branch1_3,branch2_5,branch3_3])
l2=inception_block_reduction_B(l1)
l3=residual_block_v2(l2,filters=1028)
l4=residual_block_v2(l3,filters=2056)
pool2=AveragePooling2D()(l4)
output2=Dense(n_classes,activation='softmax')(pool2)


model=Model(inputs=input,outputs=[output1,output2])
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

"""
For training the model:
(xtrain,ytrain),(xtest,ytest)=load_data()

#building the input vector 
Y_train = np_utils.to_categorical(ytrain, output_images)
Y_test = np_utils.to_categorical(ytest, output_images)
X_train=X_train.reshape(-1,264,264,3)
X_test=X_test.reshape(-1,264,264,3)
# normalizing the data to help with the training
X_train =xtrain/255
X_test = xtest/255
# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
print("Train matrix shape", Y_train.shape)
print("Test matrix shape", Y_test.shape)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Hyperparameters
EPOCHS=100
BATCH_SIZE=128
classifier=model.fit(X_train,Y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,validation_data=(X_test,Y_test)inception_v4
"""

model.save("your_file_path/MyOwnArchitecture.h5",save_format='h5')