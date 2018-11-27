from keras.utils import np_utils
from keras.models import model_from_json
from keras.datasets import cifar10
import keras
import numpy as np
from tictoc import ticktock

timer = ticktock()
timer.click()

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


# Le o arquivo json e cria o modelo

json_file = open('cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Carrega os weights no modelo

loaded_model.load_weights("cnn_pesos.h5")


# Compila o modelo
opt_stochastic = keras.optimizers.Adam(lr=0.001, decay=1e-6)
print("Compilando:")
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt_stochastic, metrics=['accuracy'])
#Resultados
scores = loaded_model.evaluate(x_test, y_test, batch_size=128, verbose=1)


#Arquitetura do modelo
loaded_model.summary()


# Confusion matrix resultado
from sklearn.metrics import confusion_matrix

Y_pred = loaded_model.predict(x_test, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

class_teste = []
for ix in range(10):
    print(ix, confusion_matrix(np.argmax(y_test, axis=1), y_pred)[ix].sum())
    class_teste.append( confusion_matrix(np.argmax(y_test, axis=1), y_pred)[ix].sum())

cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

print(cm)

cm_prop = np.true_divide(cm, cm.sum(axis=1, keepdims=True))

print(cm_prop)

print("foi")

# Codigo para montar um heatmap da matriz de confusao

import seaborn as sn
import pandas  as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
df_cm = pd.DataFrame(cm_prop,
                     labels,
                     labels)

plt.figure(figsize=(16, 10))
sn.set(font_scale=2.2)
plt.rcParams.update({'font.size': 24})

fmt = lambda x, pos: '{:.1%}'.format(x)

ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, fmt='.1%',
                cbar_kws={'format': FuncFormatter(fmt),
                          "ticks": [0, .25, .5, .75, 1]},
                linewidths=.5, cmap="Spectral_r")



ax.set_yticklabels(labels, rotation=0)
plt.title("Matriz de Confusão")
plt.ylabel("Verdadeiro")
plt.xlabel("Predito")

figure = ax.get_figure()
figure.savefig('plots/confusion_matrix.png', dpi=400)

timer.stop()
hour, min, sec = timer.getTime()
print("Tempo: " + str(hour) + "h" + str(min) + "min" + str(sec) + "s")


#temp_img=load_img("./Image/automobile1.jpg",target_size=(32,32))


print('\nTeste: %.3f Loss: %.3f' % (scores[1] * 100, scores[0]))
plt.show()


