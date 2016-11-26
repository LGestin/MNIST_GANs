from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Deconvolution2D, Activation, BatchNormalization, UpSampling2D, MaxPooling2D, Flatten, Reshape, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from math import floor

#Problem parameters
code_dim = 100

#Construction of the generator
generator = Sequential()
generator.add(Dense(10 * 13 * 13, input_dim = code_dim, init = 'glorot_normal'))
generator.add(Activation('relu'))
#generator.add(Dense(32 * 7 * 7))
#generator.add(BatchNormalization(mode = 1))
#generator.add(Activation('relu'))
generator.add(Reshape((10, 13, 13), input_shape = (32*7*7,)))
generator.add(UpSampling2D(size = (2,2)))
generator.add(Deconvolution2D(16, 3, 3, border_mode = 'valid', output_shape = (None, 16, 28, 28), input_shape = (None, 10, 26, 26)))
#generator.add(BatchNormalization(mode = 1))
#generator.add(Activation('relu'))
#generator.add(Deconvolution2D(16,3,3, border_mode='same', output_shape = (None, 16, 28, 28), input_shape = (None, 16, 28, 28)))
#generator.add(BatchNormalization(mode = 1))
generator.add(Activation('relu'))
#generator.add(Convolution2D(16, 5, 5, border_mode = 'same', init = 'glorot_uniform'))
#generator.add(BatchNormalization(mode = 1))
#generator.add(Activation('relu'))
#generator.add(UpSampling2D(size = (2,2)))
generator.add(Convolution2D(1, 3, 3, border_mode = 'same', init = 'glorot_uniform'))
generator.add(Activation('tanh'))

gsgd = SGD(lr = 0.0002, decay = 1e-6, momentum = 0.9, nesterov = True)
gadam = Adam(lr = 0.0002, beta_1 = 0.5)
generator.compile(loss='binary_crossentropy', optimizer = gadam)
print("GENERATOR: ")
generator.summary()
print()

#Construction of the discriminator
discriminator = Sequential()
discriminator.add(Convolution2D(8,5,5, border_mode = 'same', input_shape = (1,28,28), subsample = (2,2)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.35))
#discriminator.add(Convolution2D(64, 5, 5, subsample = (2,2)))
#discriminator.add(BatchNormalization(mode = 2))
#discriminator.add(LeakyReLU(0.2))
#discriminator.add(Dropout(0.35))
#discriminator.add(MaxPooling2D(pool_size = (2,2)))
discriminator.add(Convolution2D(16, 3, 3, border_mode = 'valid'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
#discriminator.add(Dense(512))
#discriminator.add(BatchNormalization(mode = 2))
#discriminator.add(LeakyReLU(0.2))
#discriminator.add(Dropout(0.35))
discriminator.add(Dense(2))
discriminator.add(Activation('softmax'))

dsgd = SGD(lr = 0.0005, decay = 1e-6, momentum = 0.9, nesterov = True)
dadam = Adam(lr = 0.0002)
discriminator.compile(loss = 'categorical_crossentropy',  optimizer = dadam)
print("DISCRIMINATOR: ")
discriminator.summary()
print()

#Function to freeze discriminator's parameter when generator is beign trained
def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

make_trainable(discriminator, False)

#Stacking up the two models to build the actual Generative Adversarial Network
GAN = Sequential()
GAN.add(generator)
GAN.add(discriminator)

sgd = SGD(lr = 0.0005, decay = 1e-6, momentum = 0.9, nesterov = True)
adam = Adam(lr = 0.0002, beta_1 = 0.5)
GAN.compile(loss = 'categorical_crossentropy', optimizer = adam)
print("GAN: ")
GAN.summary()
print()

#Data loading and preprocessing
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

X_train -= 127.5
X_test -= 127.5

X_train = X_train.astype(np.float32)/127.5
X_test = X_test.astype(np.float32)/127.5

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

#Number of epochs here stands more for number of batch than number of actual epochs
nb_epochs = 15000
batch_size = 64

#Initialize lists where losses will be stored
GAN_loss = list()
Disc_loss = list()


#pre-train discriminator
code = np.random.uniform(-1,1, size = (5000,code_dim))
gen_im = generator.predict(code)
ntrain = 5000
trainidx = np.random.randint(0,X_train.shape[0], ntrain)
Xt = X_train[trainidx,:,:,:]
X = np.vstack((gen_im, Xt))
y = np.zeros((2*ntrain,2))
y[0:ntrain,::] = [1.0, 0.0]
y[ntrain:,::] = [0.0, 1.0]
make_trainable(discriminator, True)
h = discriminator.fit(X,y, batch_size = 6*128, nb_epoch = 2, shuffle = True)

#Train the actual model
for n in range(nb_epochs):

	#Decrease learning rate after a certain time
	if n == 10000:
		gadam.lr.set_value(1e-6)
		dadam.lr.set_value(1e-5)
		adam.lr.set_value(1e-6)
		batch_size *= 2 #Experiment

	ydbatch = np.empty((batch_size,2))

	#train discriminator
	Xdbatch = np.empty((batch_size,1,28,28))
	Xgbatch = np.empty((floor(batch_size/2), code_dim))

	#This phase could have been written in a vectorized way to gain performance
	for i in range(batch_size):
		if (i%2 == 0):
			code = np.random.uniform(-1,1, size = (1,code_dim))
			X = generator.predict(code)
			y = np.array([1.0, 0.0])
			code = np.random.uniform(-1,1, size = (1,code_dim))
			Xgbatch[floor(i/2),::] = code
		else:
			r = np.random.random() * len(X_train)
			X = X_train[floor(r)]
			y = np.array([0.0, 1.0])

		Xdbatch[i] = X
		ydbatch[i] = y

	ygbatch = np.zeros((floor(batch_size/2),2))
	ygbatch[:,1] = 1.0

	make_trainable(discriminator, True)
	h = discriminator.train_on_batch(Xdbatch[0:floor(batch_size/2)], ydbatch[0:floor(batch_size/2)])
	make_trainable(discriminator, False)
	Disc_loss.append(h)

	#else:
	#train generator
	if np.random.random(1) < 0.5 + 0.2 * (n / nb_epochs):
		h = GAN.train_on_batch(Xgbatch, ygbatch)
		GAN_loss.append(h)

	print(str(n) + "/" + str(nb_epochs), end = '\r')

	if(n%2500 == 2499):
		plt.figure()
		image = generator.predict(code)[0][0]* 127.5 + 127.5
		plt.imshow(image, cmap = 'Greys_r')
		Image.fromarray(image.astype(np.uint8)).save(str(n)+".png")


#Select the images tha fool the discriminator better than some threshold
def show_n_good_gen_im(n, thres):
	codes = np.empty((n, code_dim))
	for i in range(n):
		min = 1
		code = np.random.uniform(-1, 1, size = (1, 100))
		p = GAN.predict(code)[0][0]
		while p > thres:
			code = np.random.uniform(-1, 1, size = (1, 100))
			if p < min:
				min = p
				print(min, end = '\r')
			p =GAN.predict(code)[0][0]

		plt.figure()
		plt.imshow(generator.predict(code)[0][0] * 127.5 + 127.5, cmap = 'Greys_r')
		codes[i, ::] = code
	return np.asarray(codes)

