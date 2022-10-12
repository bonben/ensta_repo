# Lab Session 1 

The objectives of this first lab session are the following:
- Familiarize yourself with pytorch
- Train a model from scratch using a state of the art architecture

We will perform all experiments on MNIST and CIFAR10 dataset. 

---
## Part 1

Familiarize yourself with pytorch by doing the [Pytorch_tutorial.ipynb](Pytorch_tutorial.ipynb) on Google Colab.

---
## Part 2 - MNIST
The following code can be used to obtain a DataLoader for MNIST, ready for training in pytorch : 

```python
from torchvision.datasets import MNIST
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader

## Normalization adapted for MNIST
normalization = transforms.Normalize((0.1302,), (0.3069,))
trans = transforms.Compose([transforms.ToTensor(),normalization])

### The data from MNIST will be downloaded in the following folder
rootdir = './data'

mnist_train = MNIST(rootdir,train=True,download='MNIST' not in os.listdir(rootdir),transform=trans)
mnist_test = MNIST(rootdir,train=False,download='MNIST' not in os.listdir(rootdir),transform=trans)

trainloader = DataLoader(mnist_train,batch_size=32,shuffle=True)
testloader = DataLoader(mnist_test,batch_size=32,  shuffle=False) 
```
In [models_mnist](models_mnist), you will find a LeNet that is enough to achieve around 99\% accuracy on MNIST. At each new training epoch, save the model if the test accuracy is better than the last ones. You must be able to reload this model afterwards, if the training is interrupted.

---
## Part 3 - CIFAR

The following code can be used to obtain a DataLoader for CIFAR10, ready for training in pytorch : 

```python
from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 will be downloaded in the following folder
rootdir = './data'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32) 
```

However, this will load the entire CIFAR10 dataset, which has 50000 examples per class for training ; this can result in a relatively long training. As a consequence, we encourage you to use the following code, with a [RandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.RandomSampler) in order to use a subset of training : 


```python
## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)
### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.
```

We will now define a state of the art deep model and train it from scratch. Check out [here](https://github.com/kuangliu/pytorch-cifar/tree/master/models) for reference implementations of modern deep models for CIFAR10. 

Choose a model among the following ones : 
- ResNet
- PreActResNet
- DenseNet
- VGG
  
Next, train it on a subset of CIFAR10. Try to compare with the performances on the full CIFAR10 [reported here](https://github.com/kuangliu/pytorch-cifar). At each new training epoch, save the model if the test accuracy is better than the last ones. You must be able to reload this model afterwards, if the training is interrupted.

A few hints : 
- Learning rate is a very important (if not the most important) hyperparameter, and is routinely scheduled to change a few times during training. A typical strategy is to divide it by 10 when reaching a plateau in performance. 
- Be careful with overfitting, which happens when the gap between Train and Test accuracy keeps getting larger. 
- Think about plotting and saving your results, so as not to lose track of your experiments. 

