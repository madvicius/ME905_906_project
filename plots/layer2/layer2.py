from keras.datasets import cifar10
from tictoc import ticktock
import numpy as np
from keras.utils import np_utils

timer = ticktock()
timer.click()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test, x_valid =  x_test[0:5000,:,:,:] , x_test[5000:10001,:,:,:]
y_test, y_valid =  y_test[0:5000,:] , y_test[5000:10001,:]


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_valid = x_valid.astype('float32')



mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)
x_valid = (x_valid - mean) / (std + 1e-7)




num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
y_valid = np_utils.to_categorical(y_valid, num_classes)

import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler

weight_decay = 1e-4


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003
    return lrate


# Função que cria o modelo
def create_model(input_shape, weight_decay):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model


model = create_model(x_train.shape[1:], weight_decay)

model.summary()

opt_stochastic = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt_stochastic,
              metrics=['accuracy'])

print("Modelo Compilado")

datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)

datagen.fit(x_train)

batch_size = 64

train_acc = []
val_acc = []
time = []

model_time = ticktock()
for i in range(10):

    model_time.click()

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size, verbose=1,
                                  validation_data=(x_valid, y_valid), epochs=1,
                                  callbacks=[LearningRateScheduler(lr_schedule)])
    train_acc.append(history.history['acc'][0])
    val_acc.append(history.history['val_acc'][0])
    print(train_acc,val_acc)

    time.append(model_time.getNow())
    model_time.reset()
    print("Foram %d" %i )


import pandas as pd

df_accu = pd.DataFrame({'train' : train_acc,
                        'val' : val_acc,
                        'time' : time
                        })
df_accu.to_csv("crossval_accu2.csv")

timer.stop()
hour, min, sec = timer.getTime()
print("Tempo: " + str(hour) + "h" + str(min) + "min" + str(sec) + "s")
