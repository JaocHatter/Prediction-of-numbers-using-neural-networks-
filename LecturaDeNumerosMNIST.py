import keras
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense,Dropout
#Con esta libreria ademàs de poder ver las imagenes en el dataset , veremos la funcion del error
import matplotlib.pyplot as plt
import seaborn as sns
#importemos el dataset
from keras.datasets import mnist
#importemos el dataset y los metadatos (caracteristicas, indices y demas materiales que ayudan a navegar en la data!)
np.random.seed(0)
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
print("Dimensiones de las entradas en nuestro data_set",xtrain.shape)
print("Dimensiones del objetivo:",ytrain.shape)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#lets see the images
#genero una interfaz con 10 figuras, cada subplot  tendra un tamaño de 20x20
f,ax=plt.subplots(1,10,figsize=(20,20))
for i in range(10):
    #al principio me mareo...
    #esta linea hace que encuentres la primera imagen con label numero "i"
    sample=xtrain[ytrain==i][0]
    ax[i].imshow(sample,cmap="gray")
    ax[i].set_title("Label: {}".format(i),fontsize=16)
plt.show()
#Ahora en el conjunto de salidas realizaremos One hot encoding, es decir, creareamos una matriz con 10 columnas, 
#en la que el numero actual de y se convierta en un 1 de la posicion <numero actual>
print("Antes:")
print(ytrain[:5])
ytrain=keras.utils.to_categorical(ytrain,10)
ytest=keras.utils.to_categorical(ytest,10)
print(ytrain[:5])
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#normalizemos los datos!
#como en las imagenes vemos que cada pixel està en la escala de 0-255,dividimos cada pixel entre 255 para tenerlos en el rango de 0-1
xtest,xtrain=xtest/255,xtrain/255

#Ahora...
#Sabemos que nuestras imagenes son de 28x28. lo que haremos es darle otra forma a estas matrices, 
# para que sean de la forma 1*784...
#de esta forma podremos ingresar el valor de cada pixel a la red neronal!
print("shape:",xtrain.shape)
xtrain=xtrain.reshape((60000,784))
xtest=xtest.reshape((10000,784))
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

#Hora de generar los modelos!

#usaremos el modelo secuencial
NNmodel=Sequential()
#Agregamos una capa densa
#esta capa densa tendrà 128 neuronas(modificable), recibe 784 entradas y la funcion de activacion una vez se multipliquen 
#las entradas con los pesos y sean sumadas con el bias serà ->ReLU<-
NNmodel.add(Dense(units=64,input_shape=(784,),activation="relu"))
NNmodel.add(Dense(units=64,activation="relu"))
#esta funcion Dropout ayuda a prevenir el sobreajuste o overfitting!
#"apaga" algunas neuronas aleatoriamente para evitar el sobreajuste
NNmodel.add(Dropout(0.25))
#Agregamos la capa de salida Xd
NNmodel.add(Dense(units=10,activation="softmax"))
#Configuremos cual serà la funcion de perdida que usemos, asì como el algoritmo de optimizacion...
NNmodel.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
#veamos todos los detalles...
NNmodel.summary()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Iniciamos el entrenamiento!

#Establezcamos el batch o la cantidad de datos que agregaremos a cada parcela de nuestro data set (se dividira nuestro dataset)
batchsize=512
#Esto quiere decir que evaluaremos 512 imagenes en nuestra red neuronal por epoca
epochs=20
NNmodel.fit(x=xtrain,y=ytrain,batch_size=batchsize,epochs=epochs)
#Una vez entrenado nuestro modelo, evaluemos con las 10 000 imagenes de prueba
testloss,testPrecision=NNmodel.evaluate(x=xtest,y=ytest)
print("Perdida: {}".format(testloss))
print("Presicion: {}".format(testPrecision))
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Mañana continuo with this xd
#generemos las pruebas
y_pred=NNmodel.predict(xtest)
y_pred=np.argmax(y_pred,axis=1)
y_constantes=np.argmax(ytest,axis=1)
print("Reales VS Prediciones")
error=0
for i in range(100):
    print("REAL: {}\tPREDICCION: {}".format(y_pred[i],y_constantes[i]))
    if y_pred[i]!=y_constantes[i]:
        error=i
#veamos un error
x_error=xtest[error]
x_error=x_error.reshape(28,28)
plt.imshow(x_error,cmap="gray")
plt.title("REAL: {}    PREDICTION: {}".format(y_constantes[error],y_pred[error]),fontsize=16)
plt.show() 