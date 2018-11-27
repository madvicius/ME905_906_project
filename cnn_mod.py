from keras.datasets import cifar10
from tictoc import ticktock
import numpy as np
from keras.utils import np_utils



#Inicia o cronometro (ler arquivo tictoc.py
timer = ticktock()
timer.click()


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#Configura a GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.60
set_session(tf.Session(config=config))



# Importa o banco de dados do Keras, ja vem separado em teste e treino
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Separar metade das imagens de teste em :teste e validacao
x_test, x_valid =  x_test[0:5000,:,:,:] , x_test[5000:10001,:,:,:]
y_test, y_valid =  y_test[0:5000,:] , y_test[5000:10001,:]



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_valid = x_valid.astype('float32')


# Padroniza todos os dados em torno da media dos dados de treino e divide todos pelo desvio padrao do treino
mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)
x_valid = (x_valid - mean) / (std + 1e-7)



#transforma as respostas em dummy
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

# Função para definir uma rotina para a taxa de aprendizado
def lr_schedule(epoch):

    lrate = 0.001
    if epoch > 40:
        lrate = 0.0008
    if epoch > 50:
        lrate = 0.0006
    if epoch > 70:
        lrate = 0.0004
    if epoch > 90:
        lrate = 0.00025
    return lrate


#Criação do modelo
model = Sequential()

#

# Primeira camada de Convolucao
model.add(Conv2D(32, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape=x_train.shape[1:]))
model.add(Activation('elu'))

# Realiza a batch normalization
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(64, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))


model.add(Conv2D(128, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

opt_stochastic = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt_stochastic,
              metrics=['accuracy'])

print("Modelo Compilado")


datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

datagen.fit(x_train)

batch_size = 100
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=x_train.shape[0] // batch_size, verbose=2,
                              validation_data=(x_valid, y_valid), epochs=100,
                              callbacks=[LearningRateScheduler(lr_schedule)])

model_json = model.to_json()
with open('cnn.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('cnn_pesos.h5')

print(history.history)

scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTeste: %.3f Loss: %.3f' % (scores[1] * 100, scores[0]))


import pandas as pd
df_history = pd.DataFrame({
    'acc': history.history['acc'][:],
    'val_acc' : history.history['val_acc'][:],

    'loss': history.history['loss'][:],
    'val_loss': history.history['val_loss'][:]

})

df_history.to_csv("cnn_epoch.csv")
timer.stop()
hour, min, sec = timer.getTime()
print("Tempo: " + str(hour) + "h" + str(min) + "min" + str(sec) + "s")

