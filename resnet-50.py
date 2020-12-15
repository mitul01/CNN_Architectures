import numpy as np
from keras import layers
from keras.layers import Input, Add,Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.layers.merge import concatenate,add
from cnn_blocks import residual_block_v2

INPUT_SHAPE=(224,224,3)
n_classes=10
input=Input(INPUT_SHAPE)

def stem(input):
	""" 
	A stem is the starting point of any CNN arch. It's output is followed by various inception modules.
	
	"""  
	l1=Conv2D(64,(7,7),strides=(2,2),activation='relu')(input)
	output=MaxPooling2D((3,3),strides=(2,2))(l1)

	return output

stem_output=stem(input)
# 3 residual blocks with filters=64
R1_1= residual_block_v2(stem_output,filters=64)
R1_2= residual_block_v2(R1_1,filters=64)
R1_3= residual_block_v2(R1_2,filters=64)
# 4 residual blocks with filters=128
R2_1= residual_block_v2(R1_3,filters=128)
R2_2= residual_block_v2(R2_1,filters=128)
R2_3= residual_block_v2(R2_2,filters=128)
R2_4= residual_block_v2(R2_3,filters=128)
# 6 residual blocks with filters=256
R3_1= residual_block_v2(R2_4,filters=256)
R3_2= residual_block_v2(R3_1,filters=256)
R3_3= residual_block_v2(R3_2,filters=256)
R3_4= residual_block_v2(R3_3,filters=256)
R3_5= residual_block_v2(R3_4,filters=256)
R3_6= residual_block_v2(R3_5,filters=256)
# 3 residual blocks with filters=512
R4_1= residual_block_v2(R3_6,filters=512)
R4_2= residual_block_v2(R4_1,filters=512)
R4_3= residual_block_v2(R4_2,filters=512)

pool=AveragePooling2D((1,1))(R4_3)

output=Dense(n_classes,activation='softmax')(pool)

model_resnet50=Model(inputs=input,outputs=output)
model_resnet50.summary()

model_resnet50.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

"""
For training the model:
(xtrain,ytrain),(xtest,ytest)=load_data()

#building the input vector 
Y_train = np_utils.to_categorical(ytrain, output_images)
Y_test = np_utils.to_categorical(ytest, output_images)
X_train=X_train.reshape(-1,224,224,3)
X_test=X_test.reshape(-1,224,224,3)
# normalizing the data to help with the training
X_train =xtrain/255
X_test = xtest/255
# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
print("Train matrix shape", Y_train.shape)
print("Test matrix shape", Y_test.shape)

model_resnet50.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Hyperparameters
EPOCHS=100
BATCH_SIZE=128
classifier=model_resnet50.fit(X_train,Y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,validation_data=(X_test,Y_test))
"""

model_resnet50.save("your_file_path/model_resnet50.h5",save_format='h5')

