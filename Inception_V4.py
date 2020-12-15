import numpy as np
from keras import layers
from keras.layers import Input, Add,Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D,Dropout
from keras.models import Model
from keras.layers.merge import concatenate,add
from cnn_blocks import inception_block_A,inception_block_B,inception_block_C
from cnn_blocks import inception_block_reduction_A,inception_block_reduction_B

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
stem_out=stem(input)
A1=inception_block_A(stem_out)
A2=inception_block_A(A1)
A3=inception_block_A(A2)
A4=inception_block_A(A3)
RA1=inception_block_reduction_A(A4)
B1=inception_block_B(RA1)
B2=inception_block_B(B1)
B3=inception_block_B(B2)
B4=inception_block_B(B3)
B5=inception_block_B(B4)
B6=inception_block_B(B5)
B7=inception_block_B(B6)
RC1=inception_block_B(B7)
C1=inception_block_C(RC1)
C2=inception_block_C(C1)
C3=inception_block_C(C2)
pool=AveragePooling2D()(C3)
flat=Flatten()(pool)
FC1=Dense(4096,activation='relu')(flat)
FC2=Dropout(0.8)(FC1)
output=Dense(n_classes,activation='softmax')(FC2)

model_inception_v4=Model(inputs=input,outputs=output)
model_inception_v4.summary()


model_inception_v4.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

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

model_inception_v4.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Hyperparameters
EPOCHS=100
BATCH_SIZE=128
classifier=model_inception_v4.fit(X_train,Y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,validation_data=(X_test,Y_test)inception_v4
"""

model_inception_v4.save("your_file_path/model_inception_v4.h5",save_format='h5')




