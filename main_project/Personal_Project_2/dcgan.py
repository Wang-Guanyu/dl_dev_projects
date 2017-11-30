from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import glob
import cv2
import matplotlib.pyplot as plt

# Data parameters
d_steps = 3
g_steps = 3

def generator_model():
    model = Sequential()
    
#    model.add(Dense(4608, input_dim=100))
#    model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    model.add(Reshape((3, 3, 512)))
#    model.add(Conv2DTranspose(256, (5,5), strides=2, padding='same',
#                              kernel_initializer=initializers.RandomNormal(stddev=0.02)))
#    model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    model.add(Conv2DTranspose(128, (5,5), strides=2, padding='same',
#                              kernel_initializer=initializers.RandomNormal(stddev=0.02)))
#    model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    model.add(Conv2DTranspose(64, (5,5), strides=2, padding='same',
#                              kernel_initializer=initializers.RandomNormal(stddev=0.02)))
#    model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    model.add(Conv2DTranspose(1, (5,5), strides=2, padding='same',
#                              kernel_initializer=initializers.RandomNormal(stddev=0.02)))
#    model.add(Activation('tanh'))

    model.add(Dense(7*7*50, input_dim=100))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 50)))
    model.add(Conv2DTranspose(20, (5,5), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Conv2DTranspose(1, (5,5), strides=2, padding='same'))
    model.add(Activation('tanh'))
    
#    model.add(Dense(7*7*50, input_dim=100))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Reshape((7, 7, 50)))
#    model.add(Conv2DTranspose(20, (5,5), strides=2, padding='same'))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Conv2DTranspose(1, (5,5), strides=2, padding='same'))
#    model.add(Activation('relu'))
    
    model.summary()
    return model


def discriminator_model():
    model = Sequential()
#    model.add(Conv2D(64, (5, 5), strides=2, padding='same', input_shape=(48, 48, 1),
#                     kernel_initializer=initializers.TruncatedNormal(stddev=0.02)))
#    model.add(Activation('tanh'))
#    
#    model.add(Conv2D(128, (5, 5), strides=2, padding='same',
#                     kernel_initializer=initializers.TruncatedNormal(stddev=0.02)))
#    model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    
#    model.add(Conv2D(256, (5, 5), strides=2, padding='same',
#                     kernel_initializer=initializers.TruncatedNormal(stddev=0.02)))
#    model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    
#    model.add(Conv2D(512, (5, 5), strides=2, padding='same',
#                     kernel_initializer=initializers.TruncatedNormal(stddev=0.02)))
#    model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    
#    model.add(Flatten())
#    model.add(Dense(1))
#    model.add(Activation('sigmoid'))
    
    model.add(Conv2D(20, (5, 5), strides=2, padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(Conv2D(50, (5, 5), strides=2, padding='same'))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
#    model.add(Conv2D(20, (5, 5), strides=2, padding='same', input_shape=(28, 28, 1)))
#    model.add(LeakyReLU(0.2))
#    model.add(Conv2D(50, (5, 5), strides=2, padding='same'))
#    model.add(LeakyReLU(0.2))
#    model.add(Flatten())
#    model.add(Dense(500))
#    model.add(LeakyReLU(0.2))
#    model.add(Dense(1))
#    model.add(Activation('sigmoid'))
    
    model.summary()
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()
#    X_train = (X_train.astype(np.float32) - 127.5)/127.5
#    X_train = X_train[:, :, :, None]
#    X_test = X_test[:, :, :, None]
    
    img_list = []
    for img in glob.glob('IR_new_data/*.png'):
        input_img = cv2.imread(img,0)
        input_img = cv2.resize(input_img,(28,28))
        img_list.append(input_img)
    X_train = np.array(img_list)
    X_train = (X_train.reshape(-1, 28, 28, 1).astype(np.float32)-127.5) / 127.5

    print(X_train.shape)
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(1000):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
