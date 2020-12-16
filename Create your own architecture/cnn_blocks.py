# All the necesaay blocks required for CNN architecture creation implemented using Keras API 
from keras.layers import Input, Add,Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.merge import concatenate,add
############  VGG Block #######################################################################

def VGG_block(previous_layer,filters,no_of_conv_layers):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters= No.of filters to be added in Conv layer 
      no_of_conv_layers = How many layers to be added in one conv block of VGG (usually 2 or 3)
  Output:
      returns the last layer of the constructed VGG block 
  """
  for _ in range(no_of_conv_layers):
    previous_layer=Conv2D(filters,kernel_size=(3,3),activation='relu')(previous_layer)
  # output of every conv block in MaxPooled to reduce it's size
  output_layer=MaxPooling2D((2,2),strides=(2,2))(previous_layer)

  return output_layer

 ############  Inception Blocks ###############################################################

def inception_block_naive(previous_layer,filters1,filters3,filters5):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters1= No.of filters to be added in Conv layer with kernel (1,1)
      filters3= No.of filters to be added in Conv layer with kernel (3,3)
      filters5= No.of filters to be added in Conv layer with kernel (5,5)
  Output:
      returns the last layer of the constructed inception block 
  """
  #1x1 layer
  conv1=Conv2D(filters1,(1,1),padding='same',activation='relu')(previous_layer)
  # 3x3 layer
  conv3=Conv2D(filters3,(3,3),padding='same',activation='relu')(previous_layer)
  # 5x5 layer
  conv5=Conv2D(filters5,(5,5),padding='same',activation='relu')(previous_layer)
  # Pool layer
  pool=MaxPooling2D((3,3),strides=(1,1),padding='same')(previous_layer)
  # connect all  4 layers from with the previous layer
  output_layer=concatenate([conv1,conv3,conv5,pool])

  return output_layer

def inception_block(previous_layer,filters1=64,filters3_in=96,filters3_out=128,filters5_in=16,filters5_out=32,filters_pool=32):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters1= No.of filters to be added in Conv layer with kernel (1,1)
      filters3_in= No.of filters to be added in input(1st) Conv layer (with kernel 1x1) of 3x3 block 
      filters3_out= No.of filters to be added in output(2nd) Conv layer (with kernel 3x3) of 3x3 block
      filters5_in= No.of filters to be added input(1st) Conv layer (with kernel 1x1) of 5x5 block
      filters5_out= No.of filters to be added output(2nd) Conv layer (with kernel 5x5) of 5x5 block
      filters_pool= No.of filters to be added in Conv layer (with kernel 1x1) which is getting MaxPooled
  Output:
      returns the last layer of the constructed inception block 
  """
  #1x1 layer
  conv1=Conv2D(filters1,(1,1),padding='same',activation='relu')(previous_layer)
  # 3x3 layer
  conv3_in=Conv2D(filters3_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv3_out=Conv2D(filters3_out,(3,3),padding='same',activation='relu')(conv3_in)
  # 5x5 layer
  conv5_in=Conv2D(filters5_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv5_out=Conv2D(filters5_out,(5,5),padding='same',activation='relu')(conv5_in)
  # Pool layer
  pool=MaxPooling2D((3,3),strides=(1,1),padding='same')(previous_layer)
  conv_pool=Conv2D(filters_pool,(1,1),padding='same',activation='relu')(pool)
  # connect all  4 layers from with the previous layer
  output_layer=concatenate([conv1,conv3_out,conv5_out,conv_pool])

  return output_layer

def inception_block_A(previous_layer,filters1=96,filters3_in=64,filters3_out=96,filters5_in=64,filters5_broken=96,filters5_out=96,filters_pool=96):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters1= No.of filters to be added in Conv layer with kernel (1,1)
      filters3_in= No.of filters to be added in input(1st) Conv layer (with kernel 1x1) of 3x3 block 
      filters3_out= No.of filters to be added in output(2nd) Conv layer (with kernel 3x3) of 3x3 block
      filters5_in= No.of filters to be added Conv layer (with kernel 1x1) 
      filters5_broken= No.of filters to be added 1st Conv layer (with kernel 3x3) broken of v1 5x5 block 
      filters5_out= No.of filters to be added 2nd Conv layer (with kernel 3x3) broken of v1 5x5 block
      filters_pool= No.of filters to be added in Conv layer (with kernel 1x1) which is getting MaxPooled
  Output:
      returns the last layer of the constructed inception block 
  """
  #1x1 layer
  conv1=Conv2D(filters1,(1,1),padding='same',activation='relu')(previous_layer)
  # 3x3 layer
  conv3_in=Conv2D(filters3_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv3_out=Conv2D(filters3_out,(3,3),padding='same',activation='relu')(conv3_in)
  # 5x5 layer
  conv5_in=Conv2D(filters5_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv5_broken=Conv2D(filters5_broken,(3,3),padding='same',activation='relu')(conv5_in)
  conv5_out=Conv2D(filters5_out,(3,3),padding='same',activation='relu')(conv5_broken)
  # Pool layer
  pool=MaxPooling2D((3,3),strides=(1,1),padding='same')(previous_layer)
  conv_pool=Conv2D(filters_pool,(1,1),padding='same',activation='relu')(pool)
  # connect all  4 layers from with the previous layer
  output_layer=concatenate([conv1,conv3_out,conv5_out,conv_pool])

  return output_layer

def inception_block_B(previous_layer,filters1=384,filters_asym1_in=192,filters_asym1_1=224,filters_asym1_2=256,
                       filters_asym2_in=192,filters_asym2_1=192,filters_asym2_2=224,filters_asym2_3=224,filters_asym2_4=256,filters_pool=128):
  """
  Input: 
      previous_layer= preceding layer ,incase of first layer previous_layer=input tensor 
      filters1= No.of filters to be added in Conv layer with kernel (1,1)
      filters_asym1_in= No.of filters to be added 1st Conv layer (with kernel 1x1) 
      filters_asym1_1= No.of filters to be added 1st asym Conv layer (with kernel 1x7) 
      filters_asym1_2= No.of filters to be added 1st asym Conv layer (with kernel 7x1) 
      filters_asym2_in= No.of filters to be added 2nd asym Conv layer (with kernel 1x1) 
      filters_asym2_1= No.of filters to be added 2nd asym Conv layer (with kernel 1x7) 
      filters_asym2_2= No.of filters to be added 2nd asym Conv layer (with kernel 7x1)
      filters_asym2_3= No.of filters to be added 2nd asym Conv layer (with kernel 1x7) 
      filters_asym2_4= No.of filters to be added 2nd asym Conv layer (with kernel 7x1)  
      filters_pool= No.of filters to be added in Conv layer (with kernel 1x1) which is getting AveragePooled 
  Output:
      returns the last layer of the constructed inception block 
  """
  #1x1 layer
  conv1=Conv2D(filters1,(1,1),padding='same',activation='relu')(previous_layer)
  # Asymmetrical layer 1
  conv_asym1_in=Conv2D(filters_asym1_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv_asym1_1=Conv2D(filters_asym1_1,(1,7),padding='same',activation='relu')(conv_asym1_in)
  conv_asym1_2=Conv2D(filters_asym1_2,(1,7),padding='same',activation='relu')(conv_asym1_1)
  # Asymmetrical layer 2
  conv_asym2_in=Conv2D(filters_asym2_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv_asym2_1=Conv2D(filters_asym2_1,(1,7),padding='same',activation='relu')(conv_asym2_in)
  conv_asym2_2=Conv2D(filters_asym2_2,(7,1),padding='same',activation='relu')(conv_asym2_1)
  conv_asym2_3=Conv2D(filters_asym2_3,(1,7),padding='same',activation='relu')(conv_asym2_2)
  conv_asym2_4=Conv2D(filters_asym2_4,(7,1),padding='same',activation='relu')(conv_asym2_3)
  #pool
  pool=AveragePooling2D((3,3),strides=(1,1),padding='same')(previous_layer)
  conv_pool=Conv2D(filters_pool,(1,1),padding='same',activation='relu')(pool)
  # connect all  4 layers from with the previous layer
  output_layer=concatenate([conv1,conv_pool,conv_asym1_2,conv_asym2_4])

  return output_layer

def inception_block_C(previous_layer,filters_out=256,filters_asym1=384,
	                  filters_asym2_1=384,filters_asym2_2=448,filters_asym2_3=512):
  """ to edit 
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters_asym1= No.of filters to be added Conv layer (with kernel 1x1) 
      filters_asym2_1= No.of filters to be added 1st Conv layer (with kernel 3x3) broken of v1 5x5 block 
      filters_asym2_2= No.of filters to be added 1st Conv layer (with kernel 3x3) broken of v1 5x5 block 
      filters_asym2_3= No.of filters to be added 1st Conv layer (with kernel 3x3) broken of v1 5x5 block 
      filters_out= No.of filters to be added in last layer of every branch (all are same as all branch need to be concatenated) 
  Output:
      returns the last layer of the constructed inception block 
  """
  # 1x1 layer
  conv1=Conv2D(filters_out,(1,1),padding='same',activation='relu')(previous_layer)
  # Asymmetrical layer 1
  conv_asym1_1=Conv2D(filters_asym1,(1,1),padding='same',activation='relu')(previous_layer)
  conv3_asym1_1=Conv2D(filters_out,(1,3),padding='same',activation='relu')(conv_asym1_1)
  conv3_asym1_2=Conv2D(filters_out,(3,1),padding='same',activation='relu')(conv_asym1_1)
  # Asymmetrical layer 2
  conv_asym2_1=Conv2D(filters_asym2_1,(1,1),padding='same',activation='relu')(previous_layer)
  conv3_asym2_1=Conv2D(filters_asym2_2,(1,3),padding='same',activation='relu')(conv_asym2_1)
  conv3_asym2_2=Conv2D(filters_asym2_3,(3,1),padding='same',activation='relu')(conv3_asym2_1)
  conv3_asym2_3=Conv2D(filters_out,(1,3),padding='same',activation='relu')(conv3_asym2_2)
  conv3_asym2_4=Conv2D(filters_out,(3,1),padding='same',activation='relu')(conv3_asym2_2)
  # Pool layer
  pool=AveragePooling2D((1,1),padding='same')(previous_layer)
  conv_pool=Conv2D(filters_out,(1,1),padding='same',activation='relu')(pool)
  # connect all  4 layers from with the previous layer
  output_layer=concatenate([conv1,conv3_asym1_1,conv3_asym1_2,conv3_asym2_3,conv3_asym2_4,conv_pool])

  return output_layer

def inception_block_reduction_A(previous_layer,filters3_in=192,filters5_in=256,filters5_broken=256,filters5_out=192):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters3_in= No.of filters to be added in input(1st) Conv layer (with kernel 1x1) of 3x3 block 
      filters5_in= No.of filters to be added 1st Conv layer (with kernel 1x1) broken of v1 5x5 block 
      filters5_broken= No.of filters to be added 1st Conv layer (with kernel 3x3) broken of v1 5x5 block 
      filters5_out= No.of filters to be added 1st Conv layer (with kernel 3x3) broken of v1 5x5 block
  Output:
      returns the last layer of the constructed inception block 
  """
  # 3x3 layer
  conv3=Conv2D(filters3_in,(1,1),strides=(2,2),padding='same',activation='relu')(previous_layer)
  # Broken 5x5 into 2 3x3
  conv5_in=Conv2D(filters5_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv5_broken=Conv2D(filters5_broken,(3,3),padding='same',activation='relu')(conv5_in)
  conv5_out=Conv2D(filters5_out,(3,3),strides=(2,2),padding='same',activation='relu')(conv5_broken)
  # Pool layer
  pool=MaxPooling2D((3,3),strides=(2,2),padding='same')(previous_layer)
  # connect all  4 layers from with the previous layer
  output_layer=concatenate([conv3,conv5_out,pool])

  return output_layer

def inception_block_reduction_B(previous_layer,filters3_in=192,filters3_out=192,filters_asym_in=256,filters_asym_1=256,filters_asym_2=320,filters_asym_out=320):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters3_in= No.of filters to be added in input(1st) Conv layer (with kernel 1x1) of 3x3 block 
      filters3_out= No.of filters to be added in output(2nd) Conv layer (with kernel 3x3) of 3x3 block
      filters_asym_in= No.of filters to be added Conv layer (with kernel 1x1) 
      filters_asym_1= No.of filters to be added 1st asym Conv layer (with kernel 1x7) 
      filters_asym_2= No.of filters to be added 1st asym Conv layer (with kernel 7x1) 
      filters_asym_out= No.of filters to be added 1st asym Conv layer (with kernel 3x3)
  Output:
      returns the last layer of the constructed inception block 
  """
  # 3x3 layer
  conv3_in=Conv2D(filters3_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv3_out=Conv2D(filters3_out,(3,3),strides=(2,2),padding='same',activation='relu')(conv3_in)
  # Asymmetrical layer
  conv_asym_in=Conv2D(filters_asym_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv_asym_1=Conv2D(filters_asym_1,(1,7),padding='same',activation='relu')(conv_asym_in)
  conv_asym_2=Conv2D(filters_asym_2,(7,1),padding='same',activation='relu')(conv_asym_1)
  conv_asym_out=Conv2D(filters_asym_out,(3,3),strides=(2,2),padding='same',activation='relu')(conv_asym_2)
  # Pool layer
  pool=MaxPooling2D((3,3),strides=(2,2),padding='same')(previous_layer)
  # connect all  4 layers from with the previous layer
  output_layer=concatenate([conv3_out,conv_asym_out,pool])

  return output_layer


############  ResNet Blocks #############################################################

def residual_block_v1(previous_layer,filters):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters= No.of filters to be added in Conv layer 
  Output:
      returns the last layer of the constructed VGG block 
  """
  x=previous_layer
  if previous_layer.shape[-1] != filters: # to match the dimensions of previous layer with current layer output so that they can be added together
    x=Conv2D(filters,(1,1),padding='same',activation='relu')(previous_layer)
  conv1=Conv2D(filters,(3,3),padding='same',activation='relu')(previous_layer)
  conv2=Conv2D(filters,(3,3),padding='same',activation='linear')(conv1)

  output_layer=add([conv2,x]) # identity mapping output=f(x)+x
  output_layer=Activation('relu')(output_layer)

  return output_layer

def residual_block_v2(previous_layer,filters):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters= No.of filters to be added in Conv layer 
  Output:
      returns the last layer of the constructed VGG block 
  """
  x=previous_layer
  conv1=Conv2D(filters,(1,1),padding='same',activation='relu')(previous_layer)
  conv2=Conv2D(filters,(3,3),padding='same',activation='relu')(conv1)
  conv3=Conv2D(previous_layer.shape[-1],(1,1),padding='same',activation='linear')(conv2) 
  # filters = previous layer dimension so that we can add both the layers to perform identity mapping

  output_layer=add([conv3,x]) # identity mapping output=f(x)+x
  output_layer=Activation('relu')(output_layer)

  return output_layer


################## Inception-ResNet Blocks ###################################################

def inception_residual_block_A(previous_layer,filters1=32,filters3_in=32,filters3_out=32,filters5_in=32,filters5_broken=32,filters5_out=32):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters1= No.of filters to be added in Conv layer with kernel (1,1)
      filters3_in= No.of filters to be added in input(1st) Conv layer (with kernel 1x1) of 3x3 block 
      filters3_out= No.of filters to be added in output(2nd) Conv layer (with kernel 3x3) of 3x3 block
      filters5_in= No.of filters to be added Conv layer (with kernel 1x1) 
      filters5_broken= No.of filters to be added 1st Conv layer (with kernel 3x3) broken of v1 5x5 block 
      filters5_out= No.of filters to be added 2nd Conv layer (with kernel 3x3) broken of v1 5x5 block
  Output:
      returns the last layer of the constructed inception block 
  """
  #1x1 layer
  conv1=Conv2D(filters1,(1,1),padding='same',activation='relu')(previous_layer)
  # 3x3 layer
  conv3_in=Conv2D(filters3_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv3_out=Conv2D(filters3_out,(3,3),padding='same',activation='relu')(conv3_in)
  # 5x5 layer
  conv5_in=Conv2D(filters5_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv5_broken=Conv2D(filters5_broken,(3,3),padding='same',activation='relu')(conv5_in)
  conv5_out=Conv2D(filters5_out,(3,3),padding='same',activation='relu')(conv5_broken)
  # connect all 3 layers from with the previous layer
  inception_output_layer=concatenate([conv1,conv3_out,conv5_out])

  # making a identical mapping
  output=Conv2D(previous_layer.shape[-1],(1,1),padding='same',activation='linear')(inception_output_layer)
  output_layer=add([output,previous_layer])
  output_layer=Activation('relu')(output_layer)
  return output_layer

def inception_residual_block_B(previous_layer,filters1=128,filters_asym_in=128,filters_asym_1=128,filters_asym_2=128):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters1= No.of filters to be added in Conv layer with kernel (1,1)
      filters_asym_in= No.of filters to be added Conv layer (with kernel 1x1) 
      filters_asym_1= No.of filters to be added 1st asym Conv layer (with kernel 1x7)  
      filters_asym_2= No.of filters to be added 1st asym Conv layer (with kernel 7x1)  
  Output:
      returns the last layer of the constructed inception block 
  """
  # 1x1 layer
  conv1=Conv2D(filters1,(1,1),padding='same',activation='relu')(previous_layer)
  # Asymmetrical layer
  conv_asym_in=Conv2D(filters_asym_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv_asym_1=Conv2D(filters_asym_1,(1,7),padding='same',activation='relu')(conv_asym_in)
  conv_asym_2=Conv2D(filters_asym_2,(7,1),padding='same',activation='relu')(conv_asym_1)
  # connect all layers from with the previous layer
  inception_output_layer=concatenate([conv1,conv_asym_2])
  # making a identical mapping
  output=Conv2D(previous_layer.shape[-1],(1,1),padding='same',activation='linear')(inception_output_layer)
  output_layer=add([output,previous_layer])
  output_layer=Activation('relu')(output_layer)
  return output_layer

def inception_residual_block_C(previous_layer,filters1=192,filters_asym_in=192,filters_asym_1=192,filters_asym_2=192):
  """
  Input: 
      previous_layer= preceding layer , incase of first layer previous_layer=input tensor 
      filters1= No.of filters to be added in Conv layer with kernel (1,1)
      filters_asym_in= No.of filters to be added Conv layer (with kernel 1x1) 
      filters_asym_1= No.of filters to be added 1st asym Conv layer (with kernel 1x3) 
      filters_asym_2= No.of filters to be added 1st asym Conv layer (with kernel 3x1) 
  Output:
      returns the last layer of the constructed inception block 
  """
  # 1x1 layer
  conv1=Conv2D(filters1,(1,1),padding='same',activation='relu')(previous_layer)
  # Asymmetrical layer
  conv_asym_in=Conv2D(filters_asym_in,(1,1),padding='same',activation='relu')(previous_layer)
  conv_asym_1=Conv2D(filters_asym_1,(1,3),padding='same',activation='relu')(conv_asym_in)
  conv_asym_2=Conv2D(filters_asym_2,(3,1),padding='same',activation='relu')(conv_asym_1)
  # connect all layers from with the previous layer
  inception_output_layer=concatenate([conv1,conv_asym_2])
  # making a identical mapping
  output=Conv2D(previous_layer.shape[-1],(1,1),padding='same',activation='linear')(inception_output_layer)
  output_layer=add([output,previous_layer])
  output_layer=Activation('relu')(output_layer)
  return output_layer

###############################################################################################################################