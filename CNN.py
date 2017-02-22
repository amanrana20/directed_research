import keras.backend as K
K.set_image_dim_ordering('th')
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Convolution3D, MaxPooling3D, BatchNormalization

def model(input_shape, output_shape):
	#  Input Layer
	inp = Input(shape=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))

	#Layer 1
	l1_conv1 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(inp)
	l1_conv2 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l1_conv1)
	l1_maxpool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(l1_conv1)

	#Layer 2
	l2_conv1 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l1_maxpool1)
	l2_conv2 = Convolution3D(32, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l2_conv1)
	l2_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l2_conv2)

	#Layer 3
	l3_conv1 = Convolution3D(64, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l2_maxpool1)
	l3_conv2 = Convolution3D(64, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l3_conv1)
	l3_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l3_conv2)

	#Layer 4
	l4_conv1 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l3_maxpool1)
	l4_conv2 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l4_conv1)
	l4_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l4_conv2)

	#Layer 5
	l5_conv1 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l4_maxpool1)
	l5_conv2 = Convolution3D(128, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l5_conv1)
	l5_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l5_conv2)

	#Layer 6
	l6_conv1 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l5_maxpool1)
	l6_conv2 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l6_conv1)
	l6_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l6_conv2)

	#Layer 7
	l7_conv1 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l6_maxpool1)
	l7_conv2 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l7_conv1)
	l7_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l7_conv2)

	#Layer 8
	l8_conv1 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l7_maxpool1)
	l8_conv2 = Convolution3D(256, 3, 3, 3, border_mode = 'same',activation='relu', init='glorot_uniform')(l8_conv1)
	l8_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2))(l8_conv2)

	# Flatten
	flat = Flatten()(l8_maxpool1)
	dense1 = Dense(256, init='glorot_uniform', activation='relu')(flat)
	dense2 = Dense(32, init='glorot_uniform', activation='relu')(dense1)
	out = Dense(output_shape, activation='softmax')(dense2)

	model = Model(input=inp, output=out)
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	return model
