

class my_model:
def create_model(self,dim, weight_decay):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import regularizers

    model = Sequential()
    num_classes = 10
# Primeira camada de Convolucao
    model.add(Conv2D(32, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l1(weight_decay),
                 input_shape=dim)
    model.add(Activation('elu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l1(weight_decay),
                 input_shape=dim)
    model.add(Activation('elu'))
    model.add(Conv2D(64, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))