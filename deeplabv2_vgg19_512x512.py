from keras.models import Model
from keras.layers import Input, Conv2D, AtrousConv2D, MaxPool2D, merge, ZeroPadding2D, Dropout, UpSampling2D

input_shape = (512, 512, 3)
img_input = Input(shape=input_shape)

# block1
h = ZeroPadding2D(padding=(1,1))(img_input)
h = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='block1_conv1')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='block1_conv2')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = MaxPool2D(pool_size=(3,3), strides=(2,2))(h)

# block2
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=128, kernel_size=(3,3), activation='relu', name='block2_conv1')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=128, kernel_size=(3,3), activation='relu', name='block2_conv2')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = MaxPool2D(pool_size=(3,3), strides=(2,2))(h)

# block3
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=256, kernel_size=(3,3), activation='relu', name='block3_conv1')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=256, kernel_size=(3,3), activation='relu', name='block3_conv2')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=256, kernel_size=(3,3), activation='relu', name='block3_conv3')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=256, kernel_size=(3,3), activation='relu', name='block3_conv4')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = MaxPool2D(pool_size=(3,3), strides=(2,2))(h)

# block4
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=512, kernel_size=(3,3), activation='relu', name='block4_conv1')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=512, kernel_size=(3,3), activation='relu', name='block4_conv2')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=512, kernel_size=(3,3), activation='relu', name='block4_conv3')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = Conv2D(filters=512, kernel_size=(3,3), activation='relu', name='block4_conv4')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = MaxPool2D(pool_size=(3,3), strides=(1,1))(h)

# block5 (start to use Atrous convolution)
h = ZeroPadding2D(padding=(2,2))(h) # padding <-> atrous_rate
h = AtrousConv2D(filters=512, atrous_rate=(2,2), kernel_size=(3,3), activation='relu', name='block5_conv1')(h)
h = ZeroPadding2D(padding=(2,2))(h)
h = AtrousConv2D(filters=512, atrous_rate=(2,2), kernel_size=(3,3), activation='relu', name='block5_conv2')(h)
h = ZeroPadding2D(padding=(2,2))(h)
h = AtrousConv2D(filters=512, atrous_rate=(2,2), kernel_size=(3,3), activation='relu', name='block5_conv3')(h)
h = ZeroPadding2D(padding=(2,2))(h)
h = AtrousConv2D(filters=512, atrous_rate=(2,2), kernel_size=(3,3), activation='relu', name='block5_conv4')(h)
h = ZeroPadding2D(padding=(1,1))(h)
h = MaxPool2D(pool_size=(3,3), strides=(1,1))(h)

# atrous (four scales)

# atrous rate = 6
b1 = ZeroPadding2D(padding=(6,6))(h) # padding <-> atrous_rate
b1 = AtrousConv2D(filters=1024, atrous_rate=(6,6), kernel_size=(3,3), activation='relu', name='fc6_scale1')(b1)
b1 = Dropout(0.5)(b1)
b1 = Conv2D(filters=1024, kernel_size=(1,1), activation='relu', name='fc7_scale1')(b1) # normal Conv2D
b1 = Dropout(0.5)(b1)
b1 = Conv2D(filters=20, kernel_size=(1,1), activation='softmax', name='fc8_scale1')(b1)# normal Conv2D

# atrous rate = 12
b2 = ZeroPadding2D(padding=(12,12))(b1)
b2 = AtrousConv2D(filters=1024, atrous_rate=(12,12), kernel_size=(3,3), activation='relu', name='fc6_scale2')(b2)
b2 = Dropout(0.5)(b2)
b2 = Conv2D(filters=1024, kernel_size=(1,1), activation='relu', name='fc7_scale2')(b2)
b2 = Dropout(0.5)(b2)
b2 = Conv2D(filters=20, kernel_size=(1,1), activation='softmax', name='fc8_scale2')(b2)
                   
# atrous rate = 18
b3 = ZeroPadding2D(padding=(18,18))(b2)
b3 = AtrousConv2D(filters=1024, atrous_rate=(18,18), kernel_size=(3,3), activation='relu', name='fc6_scale3')(b3)
b3 = Dropout(0.5)(b3)
b3 = Conv2D(filters=1024, kernel_size=(1,1), activation='relu', name='fc7_scale3')(b3)
b3 = Dropout(0.5)(b3)
b3 = Conv2D(filters=20, kernel_size=(1,1), activation='softmax', name='fc8_scale3')(b3)
                   
# atrous rate = 24
b4 = ZeroPadding2D(padding=(24,24))(b3)
b4 = AtrousConv2D(filters=1024, atrous_rate=(24,24), kernel_size=(3,3), activation='relu', name='fc6_scale4')(b4)
b4 = Dropout(0.5)(b4)
b4 = Conv2D(filters=1024, kernel_size=(1,1), activation='relu', name='fc7_scale4')(b4)
b4 = Dropout(0.5)(b4)
b4 = Conv2D(filters=20, kernel_size=(1,1), activation='softmax', name='fc8_scale4')(b4)

# merge four scales
s = merge([b1, b2, b3, b4], mode='sum')
out = UpSampling2D(size=(8,8))(s)

x = Reshape((512*512, 20))(out)