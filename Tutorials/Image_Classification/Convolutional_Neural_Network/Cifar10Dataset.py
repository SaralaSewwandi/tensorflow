from BaseDataset import BaseDataset
from tensorflow.keras import datasets

class Cifar10Dataset(BaseDataset):

 def __init__(self):
  self.class_names =  ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

 def download(self):
  pass

 def load_data(self):
  (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
  return (self.train_images, self.train_labels), (self.test_images, self.test_labels)  

 def normalize(self,factor):
  return  self.train_images / factor, self.test_images / factor
