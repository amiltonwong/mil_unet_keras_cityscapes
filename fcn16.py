input_shape = (512, 512, 3)
img_input = Input(shape=input_shape)

weight_decay = 0.01

# block1 
x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block1_conv1',
          kernel_regularizer=l2(weight_decay))(img_input)
x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block1_conv2',
          kernel_regularizer=l2(weight_decay))(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2), name='block1_pool')(x)

# block2
x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block2_conv1',
          kernel_regularizer=l2(weight_decay))(x)
x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block2_conv2',
          kernel_regularizer=l2(weight_decay))(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2), name='block2_pool')(x)

# block3
x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block3_conv1',
          kernel_regularizer=l2(weight_decay))(x)
x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block3_conv2',
          kernel_regularizer=l2(weight_decay))(x)
x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block3_conv3',
          kernel_regularizer=l2(weight_decay))(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2), name='block3_pool')(x)

# block4
x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block4_conv1',
          kernel_regularizer=l2(weight_decay))(x)
x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block4_conv2',
          kernel_regularizer=l2(weight_decay))(x)
x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block4_conv3',
          kernel_regularizer=l2(weight_decay))(x)
x4 = MaxPool2D(pool_size=(2,2), strides=(2,2), name='block4_pool')(x)

# block5
x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block5_conv1',
          kernel_regularizer=l2(weight_decay))(x4)
x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block5_conv2',
          kernel_regularizer=l2(weight_decay))(x)
x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='block5_conv3',
          kernel_regularizer=l2(weight_decay))(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2), name='block5_pool')(x)

# fully convolutional
x = Conv2D(filters=4096, kernel_size=(7,7), activation='relu', padding='same', name='fc1', 
           kernel_regularizer=l2(weight_decay))(x)
x = Dropout(0.5)(x)
x = Conv2D(filters=4096,kernel_size=(1,1), activation='relu', padding='same', name='fc2',
           kernel_regularizer=l2(weight_decay))(x)
x = Dropout(0.5)(x)

# segment
x = Conv2D(filters=20, kernel_size=(1,1), activation='softmax', padding='same',
           kernel_regularizer=l2(weight_decay))(x)

# upsample x 32
x = UpSampling2D(size=(32,32))(x)

# Add
# fully convolutional
y = Conv2D(filters=4096, kernel_size=(7,7), activation='relu', padding='same', name='fc1_', 
           kernel_regularizer=l2(weight_decay))(x4)
y = Dropout(0.5)(y)
y = Conv2D(filters=4096,kernel_size=(1,1), activation='relu', padding='same', name='fc2_',
           kernel_regularizer=l2(weight_decay))(y)
y = Dropout(0.5)(y)

# segment
y = Conv2D(filters=20, kernel_size=(1,1), activation='softmax', padding='same',
           kernel_regularizer=l2(weight_decay))(y)

# upsample x 16
y = UpSampling2D(size=(16,16))(y)

# merge
x = merge([x, y] , mode='sum')

x = Reshape((512*512, 20))(x)


# construct final model
model = Model(inputs=img_input, outputs=x)

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['acc'])