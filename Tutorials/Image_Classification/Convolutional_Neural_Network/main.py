from Cifar10Dataset import Cifar10Dataset
from Cifar10ImageVisualization import Cifar10ImageVisualization
from Cifar10Model import Cifar10Model

def main():
 cifar10_dataset=Cifar10Dataset()
 (train_images, train_labels), (test_images, test_labels) = cifar10_dataset.load_data()
 train_images, test_images = cifar10_dataset.normalize(255.0)
 #print(train_images.shape)
 #print(train_labels.shape)
 #print(test_images.shape)
 #print(test_labels.shape)
 #c10iv=Cifar10ImageVisualization()
 #c10iv.plot_images(train_images,train_labels,cifar10_dataset.class_names)
 model=Cifar10Model()
 model.build()
 # model.plot_accuracy()  
 model.summary()
 model.compile()
 model.train(train_images,train_labels,test_images,test_labels)
 model.plot_train_and_validation_accuracy()
 return
 model.save("./temp/saved_model")
 reloaded_model= model.reload("./temp/saved_model/1660382171")
 reloaded_model.evaluate(test_images,test_labels)
  

if __name__ == "__main__":
 main()
