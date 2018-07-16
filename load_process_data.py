import torch
from torchvision import datasets, transforms, models, utils
import os
import json

def load_process_data(data_dir):


    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'valid', 'test']}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                       for x in ['train', 'valid', 'test']}

    dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'valid', 'test']}

# ### Label mapping
# Import a list of category nember and corresponding flower species 

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return dataloaders, dataset_sizes, cat_to_name
