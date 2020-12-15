import keras 
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

output_images=10
activation_function='relu'

model_alexnet=Sequential([
			#1st layer
			Conv2D(input_shape=(32,32,3),filters=96,kernel_size=(6,6),strides=(1,1),activation=activation_function),
			MaxPooling2D(pool_size=(3,3),strides=(1,1)),
			BatchNormalization(),
			#2nd layer
			Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),activation=activation_function),
			MaxPooling2D(pool_size=(3,3),strides=(1,1)),
			BatchNormalization(),
			#3rd layer
			Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),activation=activation_function),
			#4th layer
			Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),activation=activation_function),
			#5th layer
			Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),activation=activation_function),
			MaxPooling2D(pool_size=(3,3),strides=(1,1)),
			Flatten(),
			#6th layer
			Dense(4096,activation=activation_function),
			Dropout(0.5),
			#7th layer
			Dense(4096,activation=activation_function),
			Dropout(0.5),
			#8th layer
			Dense(output_images,activation='softmax')
			])

print(model_alexnet.summary)
"""
Note : Values of filters, kernel size and strides change with input dimension. 
In the original AlexNet , input dimension was 227 , but we here take 32 due to which we need to reduce 
kernel size and strides. 
In original Alexnet :- 1st layer --> Kernel Size=(11,11) and strides=(4,4)
                       In Pooling layers --> stride=(2,2)
                       Other Conv layers --> filter of size (5,5) were used
"""
(xtrain,ytrain),(xtest,ytest)=keras.datasets.cifar10.load_data()

#building the input vector 
Y_train = np_utils.to_categorical(ytrain, output_images)
Y_test = np_utils.to_categorical(ytest, output_images)
X_train=X_train.reshape(-1,32,32,3)
X_test=X_test.reshape(-1,32,32,3)
# normalizing the data to help with the training
X_train =xtrain/255
X_test = xtest/255
# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
print("Train matrix shape", Y_train.shape)
print("Test matrix shape", Y_test.shape)

model_alexnet.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

classifier_cifar=model_alexnet.fit(X_train,Y_train,epochs=50,batch_size=128,verbose=1,validation_data=(X_test,Y_test))

model_alexnet.save("your_file_path/model_alexnet_cifar.h5",save_format='h5')