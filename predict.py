import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models, utils
from collections import OrderedDict
import numpy as np
import time
import os
import csv
from PIL import Image
import json
from datetime import date

from parse_predict_input import parse_predict_input

args = parse_predict_input()

device = torch.device("cuda:0" if torch.cuda.is_available() & args.gpu else "cpu")

image_path = args.path_to_image
model_path = args.path_to_checkpoint


# ## Loading the checkpoint
# 
# Function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if (checkpoint['arch'] == 'vgg16'):
        model = models.vgg16(pretrained=True)
    elif (checkpoint['arch'] == 'vgg11'):
        model = models.vgg11(pretrained=True)
    elif (checkpoint['arch'] == 'vgg13'):
        model = models.vgg13(pretrained=True)
    elif (checkpoint['arch'] == 'vgg19'):
        model = models.vgg19(pretrained=True)
    else:
        print('Model not supported. Only VGG 11,13,16,19 networks are supported')
    
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_sizes = checkpoint['hidden_layers']
    p_drop = checkpoint['p_drop']

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=p_drop)),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(p=p_drop)),
                          ('logits', nn.Linear(hidden_sizes[1], output_size)),
                          ('output', nn.LogSoftmax(dim=1))
]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
 

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
     # Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    image = image.resize((256,256))
    image = image.crop((16, 16, 256, 256))
    img = np.array(image) / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = img.transpose((2, 0, 1))

    return img


# ## Class Prediction

def predict(image_path, model_path, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = load_checkpoint(model_path)
    model = model.double()
    model = model.to(device)
    idx_to_class = {value: int(key) for key, value in model.class_to_idx.items()}
    model.eval()

    img = process_image(image_path)
    image = torch.from_numpy(img)
    image.unsqueeze_(0)
    # Convert the image to a numpy array

    # Calculate the class probabilities for img
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        
    probs, idx = torch.topk(ps, topk)
    probs = probs[0].numpy()

    classes = [idx_to_class[x] for x in idx[0].numpy()]
    
    return probs.tolist(), classes

        
# ## Call the predict function and print out the topK classes and their probabilities

probs, classes = predict(image_path, model_path, args.top_k)

for x in range(len(classes)):
    print("\n{}:   {}".format(classes[x], probs[x]))

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    i = 0
    for x in classes:
        print("\n{}:   {}".format(cat_to_name[str(x)], probs[i]))
        i = i+1

else: print("Use the option --category_names to pass a dictionary of flower species")
