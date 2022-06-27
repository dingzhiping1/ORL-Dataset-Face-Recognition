from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# dataset preparation
dataset = np.zeros((350,112,92,1))
labels = np.zeros((350,))

def show_image(filepath):
    im = Image.open(filepath)
    return np.array(im)

for i in range(4):
    for j in range(10):
        data = show_image('ORL Faces Database/s'+str(i+1)+'/'+str(j+1)+'.bmp')
        labels[i*10+j,] = i+1
        for k in range(112):
            for n in range(92):
                dataset[i*10+j,k,n,0] = data[k,n]

for i in range(5,35):
    for j in range(10):
        data = show_image('ORL Faces Database/s'+str(i+5)+'/'+str(j+1)+'.bmp')
        labels[i*10+j,] = i
        for k in range(112):
            for n in range(92):
                dataset[i*10+j,k,n,0] = data[k,n]

x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.10)
print("dataset preparation completed")

# data preprocessing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, 35)
y_test = to_categorical(y_test, 35)

# building CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(112, 92, 1)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(35, activation='softmax'))


# training compiling
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model training
history = model.fit(x_train, y_train,validation_split=0.1, validation_data=None,
          shuffle=True, batch_size=32, epochs=20)

# model evaluation
score = model.evaluate(x_test, y_test)
print('testing accuracy: ', score[1])


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
