#Made by Ithiel in 11/02/2021

from keras.datasets import boston_housing
from sklearn import preprocessing
from keras import models
from keras import layers

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(f'Training data : {train_data.shape}')
print(f'Test data : {test_data.shape}')

train_normalized = preprocessing.normalize(train_data)
test_normalized = preprocessing.normalize(test_data)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))


model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])

model.fit(train_normalized, train_targets, epochs=80, batch_size=16, verbose=0)
model.evaluate(test_normalized,test_targets)