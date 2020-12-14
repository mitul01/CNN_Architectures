import keras 
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
 
input_shape=(28,28,1)
output_images=10
activation_function='tanh'

model=Sequential([
			#1st layer
			Conv2D(input_shape=input_shape,filters=6,kernel_size=(5,5),strides=(1,1),activation=activation_function),
			AveragePooling2D(pool_size=(2,2),strides=(1,1)),
			#2nd layer
			Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),activation=activation_function),
			AveragePooling2D(pool_size=(3,3),strides=(1,1)),
			Flatten(),
			#3rd layer
			Dense(120,activation=activation_function),
			#4th layer
			Dense(32,activation=activation_function),
			#5th layer
			Dense(10,activation='softmax')
			])

print(model.summary())

(xtrain,ytrain),(xtest,ytest)=keras.datasets.mnist.load_data(path="mnist.npz")

# building the input vector from the 28x28 pixels
Y_train = np_utils.to_categorical(ytrain, output_images)
Y_test = np_utils.to_categorical(ytest, output_images)
# normalizing the data to help with the training
X_train =xtrain/255
X_test = xtest/255
# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
print("Train matrix shape", Y_train.shape)
print("Test matrix shape", Y_test.shape)
X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

classifier=model.fit(X_train,Y_train,epochs=50,batch_size=128,verbose=1,validation_data=(X_test,Y_test))