import torch
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import numpy as np

class FMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, rotate_test=False, rotate_train=False, use_augmentations=False):
        # Load FashionMNIST dataset
        self.__train = train
        self.rotate_test = rotate_test
        self.rotate_train = rotate_train
        self.use_augmentations = use_augmentations

        
        transformation_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]

        if self.use_augmentations and self.__train:
            transformation_list.insert(0, transforms.RandomRotation(degrees=(-15, 15))) 

        self.transform = transforms.Compose(transformation_list)

        # Define the restricted classes
        self.class_names = ["T-shirt/top", "Trouser", "Pullover", "Sneaker", "Bag", "Ankle Boot"]
        
        # Load the dataset
        if self.__train:
            full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
        else:
            full_dataset = datasets.FashionMNIST(root='./data', train=False, download=True)

        # Filter out unwanted classes and reassign labels
        self.data = []
        self.labels = []
        
        for img, label in full_dataset:
            if label in [0, 1, 2, 7, 8, 9]:  # Corresponding to our restricted classes
                new_label = [0, 1, 2, 3, 4, 5][[0, 1, 2, 7, 8, 9].index(label)]  
                self.data.append(img)
                self.labels.append(new_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]

        if (self.rotate_train and self.__train) or (self.rotate_test and not self.__train):  
            angle = random.choice([0, 90, 180, 270]) # Randomly select a rotation angle
            img = img.rotate(angle)

        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img) 
        
        return img, label
    
    def __str__(self):
        return "This is FMNIST Dataset."

    @property
    def is_train(self):
        return self.__train
    
    @property
    def label_dict(self):
        return {i: name for i, name in enumerate(self.class_names)}




    


if __name__=="__main__":
    fm = FMNIST()
    print(fm)
    