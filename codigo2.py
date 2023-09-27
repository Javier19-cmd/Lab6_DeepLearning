import keras
from keras.layers import Dense, Conv2DTranspose, LeakyReLU, Reshape, BatchNormalization, Activation, Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

# Función para cargar imágenes desde archivos CSV
import pandas as pd

def cargar_imagenes_desde_csv(file_path):
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv(file_path, header=None)

    # Extraer las etiquetas (si están en la primera columna)
    labels = data.iloc[:, 0].values
    data = data.iloc[:, 1:].values  # Excluir las etiquetas

    # Normalizar los datos a [-1, 1] si es necesario
    data = (data - 127.5) / 127.5  # Suponiendo que los valores están en el rango [0, 255]

    # Redimensionar los datos según las dimensiones de las imágenes de MNIST (28x28)
    data = data.reshape(-1, 28, 28, 1)  # Agregar una dimensión de canal

    return data, labels

# Función para crear el generador
def generador_de_imagenes():
    generador = Sequential()
    generador.add(Dense(128 * 7 * 7, input_shape=(100,)))
    generador.add(LeakyReLU())
    generador.add(Reshape((7, 7, 128)))
    generador.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"))
    generador.add(LeakyReLU(alpha=0.2))
    generador.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation='tanh'))
    return generador

# Función para crear el discriminador
def discriminador_de_imagenes():
    discriminador = Sequential()
    discriminador.add(Conv2D(64, kernel_size=3, padding="same", input_shape=(28, 28, 1)))
    discriminador.add(LeakyReLU(alpha=0.2))
    discriminador.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding="same"))
    discriminador.add(LeakyReLU(alpha=0.2))
    discriminador.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding="same"))
    discriminador.add(LeakyReLU(alpha=0.2))
    discriminador.add(Conv2D(256, kernel_size=3, strides=(2, 2), padding="same"))
    discriminador.add(LeakyReLU(alpha=0.2))
    discriminador.add(Flatten())
    discriminador.add(Dropout(0.4))
    discriminador.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    discriminador.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return discriminador

# Función para crear la GAN
def crear_gan(discriminador, generador):
    discriminador.trainable = False
    gan = Sequential()
    gan.add(generador)
    gan.add(discriminador)
    opt = Adam(lr=0.0002, beta_1=0.5)
    gan.compile(loss="binary_crossentropy", optimizer=opt)
    return gan

# Función para generar datos de entrada aleatorios
def generar_datos_entrada(n_muestras):
    X = np.random.randn(100 * n_muestras)
    X = X.reshape(n_muestras, 100)
    return X

# Función para crear datos falsos con el generador
def crear_datos_fake(modelo_generador, n_muestras):
    input = generar_datos_entrada(n_muestras)
    X = modelo_generador.predict(input)
    y = np.zeros((n_muestras, 1))
    return X, y

# Función para cargar datos reales aleatorios
def cargar_datos_reales(dataset, n_muestras):
    ix = np.random.randint(0, dataset.shape[0], n_muestras)
    X = dataset[ix]
    y = np.ones((n_muestras, 1))
    return X, y

# Función para entrenar el discriminador
def entrenar_discriminador(modelo, dataset, n_iteraciones=20, batch=128):
    medio_batch = int(batch / 2)

    for i in range(n_iteraciones):
        X_real, y_real = cargar_datos_reales(dataset, medio_batch)
        _, acc_real = modelo.train_on_batch(X_real, y_real)

        X_fake, y_fake = crear_datos_fake(modelo_generador, medio_batch)
        _, acc_fake = modelo.train_on_batch(X_fake, y_fake)

        print(str(i + 1) + ' Real:' + str(acc_real * 100) + ', Fake:' + str(acc_fake * 100))

# Función para mostrar imágenes generadas
def mostrar_imagenes_generadas(datos_fake, epoch):
    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    datos_fake = (datos_fake + 1) / 2.0
    for i in range(10):
        plt.imshow(datos_fake[i, :, :, 0], cmap='gray')
        plt.axis('off')
        nombre = str(epoch) + '_imagen_generada_' + str(i) + '.png'
        plt.savefig(nombre, bbox_inches='tight')
        plt.close()

# Función para evaluar y guardar el modelo generador
def evaluar_y_guardar(modelo_generador, epoch, medio_dataset):
    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    nombre = str(epoch) + '_' + str(now) + "_modelo_generador.h5"
    modelo_generador.save(nombre)
    X_real, _ = cargar_datos_reales(X_train, medio_dataset)
    X_fake, _ = crear_datos_fake(modelo_generador, medio_dataset)
    _, acc_real = discriminador.evaluate(X_real, np.ones((medio_dataset, 1)))
    _, acc_fake = discriminador.evaluate(X_fake, np.zeros((medio_dataset, 1)))
    print('Acc Real:' + str(acc_real * 100) + '% Acc Fake:' + str(acc_fake * 100) + '%')

# Función para entrenar la GAN
def entrenamiento(datos, modelo_generador, discriminador, gan, epochs, n_batch, inicio=0):
    dimension_batch = int(datos.shape[0] / n_batch)
    medio_dataset = int(n_batch / 2)

    for epoch in range(inicio, inicio + epochs):
        for batch in range(n_batch):
            X_real, y_real = cargar_datos_reales(datos, medio_dataset)
            coste_discriminador_real, _ = discriminador.train_on_batch(X_real, y_real)

            X_fake, y_fake = crear_datos_fake(modelo_generador, medio_dataset)
            coste_discriminador_fake, _ = discriminador.train_on_batch(X_fake, y_fake)

            X_gan = generar_datos_entrada(medio_dataset)
            Y_gan = np.ones((medio_dataset, 1))
            coste_gan = gan.train_on_batch(X_gan, Y_gan)

        if (epoch + 1) % 10 == 0:
            evaluar_y_guardar(modelo_generador, epoch=epoch, medio_dataset=medio_dataset)
            mostrar_imagenes_generadas(X_fake, epoch=epoch)

# Ruta de los archivos CSV de MNIST
train_file_path = 'mnist_train.csv'
test_file_path = 'mnist_test.csv'

# Cargar datos de entrenamiento
X_train, y_train = cargar_imagenes_desde_csv(train_file_path)

# Cargar datos de prueba
X_test, y_test = cargar_imagenes_desde_csv(test_file_path)

# Crear generador, discriminador y GAN
modelo_generador = generador_de_imagenes()
discriminador = discriminador_de_imagenes()
gan = crear_gan(discriminador, modelo_generador)

# Entrenar el modelo
entrenamiento(X_train, modelo_generador, discriminador, gan, epochs=100, n_batch=64)
