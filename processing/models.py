from keras import Sequential
from keras.layers import Flatten, Dense


def simpleRGB(input_shape):
    model = keras.Sequential([
                    keras.layers.Flatten(input_shape=input_shape),
                    keras.layers.Dense(input_shape[0]/2, activation='relu'),
                    keras.layers.Dense(2, activation='softmax')])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def simpleGray(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(54, 54, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

