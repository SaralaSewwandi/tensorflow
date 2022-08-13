from BaseImageVisualization import BaseImageVisualization
import matplotlib.pyplot as plt

class Cifar10ImageVisualization(BaseImageVisualization):
 def __init__(self):
  pass
 
 def plot_image(self):
  pass
 
 def plot_images(self,train_images,train_labels,class_names):
  plt.figure(figsize=(10,10))
  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
  plt.show()
  

