import tensorflow as tf
from tensorflow.keras import  models
import time

class BaseModel:
 def __init__(self):
  self.model=None
  self.input_shape = None

 def build(self):
  self.model = models.Sequential()

 def compile(self):
  self.model.compile(optimizer=None,
              loss=None,
              metrics=None)

 def train(self,train_images,train_labels,test_images, test_labels):
  pass

 def plot_train_and_validation_accuracy(self,train_history):
  plt.plot(train_history.history['accuracy'], label='accuracy')
  plt.plot(train_history.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch') 
  plt.ylabel('Accuracy')
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')
  plt.show()


 def evaluate(self,test_images,test_labels):
  test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)
  print("test accuracy", test_acc)
  print("test loss", test_loss)

 def predict(self):
  pass

 def save(self,location):
  t = time.time()
  export_path = location+"/{}".format(int(t))
  self.model.save(export_path)
  print(export_path)

 def reload(self,saved_path):
  return tf.keras.models.load_model(saved_path)
 
 def summary(self):
  self.model.summary()
