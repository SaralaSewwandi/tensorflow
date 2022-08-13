from BaseModel import BaseModel
import tensorflow as tf
from tensorflow.keras import  layers
import matplotlib.pyplot as plt

class Cifar10Model(BaseModel):
 def __init__(self, input_shape=(32,32,3)):
  super().build()
  self.input_shape = input_shape
  self.train_history = None

 def build_convolutional_base(self):
  self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
  self. model.add(layers.MaxPooling2D((2, 2)))
  self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  self.model.add(layers.MaxPooling2D((2, 2)))
  self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

 def build_classification_segment(self):
  self.model.add(layers.Flatten())
  self.model.add(layers.Dense(64, activation='relu'))
  self.model.add(layers.Dense(10))

 def build(self):
  self.build_convolutional_base()
  self.build_classification_segment()


 def compile(self):
  self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

 def train(self,train_images,train_labels,test_images, test_labels):
  self.train_history = self.model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
  return self.train_history

 def plot_train_and_validation_accuracy(self):
  super().plot_train_and_validation_accuracy(self.train_history)

 def evaluate(self,test_images,test_labels):
  super().evaluate(test_images, test_labels)
 
 def predict(self):
  pass

 def save(self,location):
  super().save(location)

 def reload(self,saved_path):
  return super().reload(saved_path)

 def summary(self):
  super().summary()
 

