

from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=1,activation="sigmoid"))


model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator
train = ImageDataGenerator(rescale=1./255,horizontal_flip=True,zoom_range=0.2)
test = ImageDataGenerator(rescale=1./255)

train_data = train.flow_from_directory('C:/Users/jegus/PycharmProjects/Practice/dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_data = test.flow_from_directory('C:/Users/jegus/PycharmProjects/Practice/dataset/test_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')


model.fit(train_data,steps_per_epoch=80,epochs=5,validation_data=test_data,validation_steps=20)

import numpy as np
from PIL import Image

test_image = Image.open('C:/Users/jegus/PycharmProjects/Practice/dataset/single_prediction/cat_or_dog_1.jpg')
test_image = test_image.resize((64, 64))
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)

if result[0][0] == 1:
    print("prediction = 'dog'")
else:
    print("prediction = 'cat'")
