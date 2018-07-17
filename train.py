import matplotlib.pyplot as plt
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
from datetime import date

from parse_train_input import parse_train_input
from load_process_data import load_process_data

# Parse the user's input
args = parse_train_input()

# Load the images from the data folder.
# The load_process_data function process images for training, validation and testing
# It also reads the flower cathegory names from a json file 
image_datasets, dataloaders, dataset_sizes, cat_to_name = load_process_data(args.data_dir)

# If available run the training on cuda
device = torch.device("cuda:0" if torch.cuda.is_available() & args.gpu else "cpu")


# Load a pretrained model
if (args.arch == 'vgg16'):
    model = models.vgg16(pretrained=True)
elif (args.arch == 'vgg11'):
    model = models.vgg11(pretrained=True)
elif (args.arch == 'vgg13'):
    model = models.vgg13(pretrained=True)
elif (args.arch == 'vgg19'):
    model = models.vgg19(pretrained=True)
else:
    print('Model not supported. Only VGG 11,13,16,19 networks are supported')
    
model = model.to(device)

# We don't need to run backpropagation on the pretrained network, only on the classifier
for param in model.parameters():
    param.requires_grad = False
    
input_size = 25088
hidden_sizes = [args.hidden_units_1, args.hidden_units_2]
output_size = len(cat_to_name)
p_drop = 0.5

# Build a feed-forward classifier
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

# Implement a function for validating the model
def validation(model, data_loader, criterion):
    valid_loss = 0
    accuracy = 0
    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy


# ## Train the classifier
criterion = nn.NLLLoss()
learn_r = args.learning_rate
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_r)
# Use the scheduler to reduce the learning rate as the training progresses
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

running_loss = {x: 0 for x in ['train', 'valid', 'test']}
data_filename = 'track_performance' + str(date.today()) + '.csv'
    
epochs = args.epochs
steps = 0
steps_to_print = 30

model.train()

for e in range(epochs):
    
    scheduler.step()
    running_loss['train'] = 0
    
    for images, labels in dataloaders['train']:

        images = images.to(device)
        labels = labels.to(device)
    
        steps +=1
        
        # clear all the gradients previously calcultated
        optimizer.zero_grad()
                     
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss['train'] += loss.item()
        
        # Run a validation test after a fixed number of training steps        
        if steps % steps_to_print == 0:
            
            model.eval()
            
            # Turn off gradients for validation
            with torch.no_grad():
                running_loss['valid'], accuracy = validation(model, dataloaders['valid'], criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss['train']/steps_to_print),
                  "Validation Loss: {:.3f}.. ".format(running_loss['train']/dataset_sizes['valid']),
                  "Validation Accuracy: {:.3f}".format(accuracy/dataset_sizes['valid']))
                
            # Write thecurrent performance to a csv file (to later plot trends)
            fields = [running_loss['train']/steps_to_print, running_loss['train']/dataset_sizes['valid'], accuracy.item()/dataset_sizes['valid']]
            with open(os.path.join(args.save_dir, data_filename), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                    
            running_loss['train'] = 0
            
            # Switch the model back to training mode
            model.train()


# ## Testing the network

# Validation on the test set
model.eval()
        
with torch.no_grad():
    running_loss['test'] = 0
    accuracy_test = 0
    
    for images, labels in dataloaders['test']:
        
        images = images.to(device)
        labels = labels.to(device)
        
        output = model.forward(images)
        running_loss['test'] += criterion(output, labels).item()
 
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy_test += equality.type(torch.FloatTensor).mean()
        
print("Test Loss: {:.3f}.. ".format(running_loss['test']/dataset_sizes['test']),
      "Test Accuracy: {:.3f}".format(accuracy_test/dataset_sizes['test']))
        

# ## Save the checkpoint

print("Model used for training: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())

model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = { 'arch': args.arch, 
               'input_size': input_size,
               'output_size': output_size,
               'hidden_layers': hidden_sizes,
               'p_drop': p_drop,
               'learn_rate': learn_r,
               'n_epochs': epochs,
               'class_to_idx': model.class_to_idx,
               'state_dict': model.state_dict()}

filename = "checkpoint-" + str(date.today()) + ".pth"
torch.save(checkpoint, os.path.join(args.save_dir, filename))


# Plot the training and validation losses to check if the model is overfitting
x, y, z = np.loadtxt(os.path.join(args.save_dir,data_filename), delimiter=',', unpack=True)

fig = plt.figure(1)
plt.subplot(121)
plt.plot(range(len(x)), x, 'k', label='Training loss')
plt.plot(range(len(y)), y, 'r', label='Validation loss')
plt.xlabel('# iterations / 30')
plt.ylabel('Loss', fontsize=14)
plt.suptitle('NN performance during training and validation', fontsize=16)
plt.legend()

plt.subplot(122)
plt.plot(range(len(z)), z, 'r', label='Validation accuracy')
plt.xlabel('# iterations / 30')
plt.ylabel('Accuracy', fontsize=14)
plt.legend()

fig.set_size_inches(w=10,h=5)
plot_filename = "NNperformance_plot-" + str(date.today()) + ".pdf"
fig.savefig( os.path.join(args.save_dir, plot_filename) )

