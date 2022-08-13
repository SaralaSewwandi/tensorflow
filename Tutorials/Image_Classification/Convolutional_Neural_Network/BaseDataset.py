class BaseDataset:
 def __init__(self):
  self.train_images=None
  self.train_labels=None
  self.test_images=None
  self.test_labels=None
  self.class_names=None


 def download(self):
  pass

 def load_data(self):
  pass

 def normalize(self,factor):
  pass
