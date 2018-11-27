

from keras.utils import np_utils
from keras.models import model_from_json
from keras.datasets import cifar10
import keras
import numpy as np
from tictoc import ticktock
from matplotlib import pyplot

timer = ticktock()
timer.click()

# Importa o bando de dados cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

fig = pyplot.figure()
pyplot.imshow(x_train[666])
fig.savefig("plots/actmap/input.png")


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')



#Normaliza os dados em torno da media de todos os pixeis
# de treinamento a 1 desvio padrao de todos os pixeis

mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)


# Transforma os dados para serem interprretados como fatores
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# Le o arquivo json e cria o modelo

json_file = open('cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Carrega os weights no modelo

loaded_model.load_weights("cnn_pesos.h5")
loaded_model.summary()


# Compila o modelo
opt_stochastic = keras.optimizers.Adam(lr=0.001, decay=1e-6)

print("Compilando:")
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt_stochastic, metrics=['accuracy'])

loaded_model.summary()

from keras.models import Model

def get_activation(layer_name,batch):
    intermediate_layer_model = Model(inputs=loaded_model.input,
                                     outputs=loaded_model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(batch)
    return intermediate_output



layer_name = ["conv2d_1","max_pooling2d_1","max_pooling2d_2","max_pooling2d_3"]

print(layer_name)

for name in layer_name:

    pic = get_activation(name,x_train)[666,:,:,15]
    fig = pyplot.figure()
    pyplot.imshow(pic,cmap="jet")
    fig.savefig("plots/actmap/"+ name + ".png")
    print("Camada: " + name + " lida")
