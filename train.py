# Imports here
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import json
import time
import os
import random

from PIL import Image


%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch import optim
import torchvision
import torch.utils.data
from torchvision import datasets, transforms, models

data_dir = './flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
# Defining the data transformations for the training and validation datasets

data_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Creating the training dataset
train_dataset = datasets.ImageFolder(data_dir + '/train', transform=data_transforms)

# Create the validation dataset
valid_dataset = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms)

# Creating the test dataset
test_dataset = datasets.ImageFolder(data_dir + '/test', transform=data_transforms)

# Creating the data loaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms)
                  for x in ['train', 'valid', 'test']}

# For variable holding names for classes
class_names = {x: image_datasets[x].classes for x in ['train', 'valid', 'test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders

# Loading Images ImageFolder and applying the transformations
# Batch size refers to the number of images used in a single pass
# Shuffle = True means there will be randomized image selection for a batch
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid', 'test']}

# Calculating accuracies of the training, validation and testing sets
dataset_sizes = {x: len(image_datasets[x])
                 for x in ['train', 'valid', 'test']}

# Code to use cuda GPU if available
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
    # TODO: Build and train your network

# Loading a pre-trained model network
model = models.vgg16(pretrained=True)
model


# Freezing the model feature parameters to avoid backpropagating through them to ensure only the
# weights of the feed-forward network (Classifier) are being updated

for param in model.parameters():
    param.requires_grad = False
    
    
    # Employing Object Oriented Programming (OOP) technique to define a new, untrained feed-forward
# network as a classifier, using ReLU activation functions and dropout

# To define a new class called Classifier that inherits from the nn.Module parent class. This class will be
# used to define a feed-forward neural network as a classifier
class Classifier(nn.Module):

    # To define the constructor method for the Classifier class. It takes in several parameters:
    # input_size (the size of the input features), output_size (the number of output classes),
    # hidden_layers (a list of sizes for the hidden layers), and drop_out (the dropout probability,
    # with a default value of 0.2).
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.2):

        # To call the constructor of the parent class (nn.Module) to initialize the Classifier object.
        super().__init__()

        # To create a list called hidden_layers to store the hidden layers of the neural network.
        # The first hidden layer is created using the nn.Linear class, which represents a fully
        # connected layer. The input size of this layer is input_size and the output size is
        # hidden_layers[0]
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # To create a zip object that pairs each element of hidden_layers with its subsequent element.
        # This will be used to define the connections between the hidden layers.
        h_layers = zip(hidden_layers[:-1], hidden_layers[1:])

        # To extend the hidden_layers list with additional hidden layers. Each hidden layer is created
        # using the nn.Linear class, with input size h_input and output size h_output, where h_input
        # and h_output are obtained from the zip object.
        self.hidden_layers.extend([nn.Linear(h_input, h_output) for h_input, h_output in h_layers])

        # To create the output layer of the neural network using the nn.Linear class. The input size
        # of this layer is the last element of hidden_layers, and the output size is output_size
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # To create a dropout module with a dropout probability of drop_out. Dropout is a
        # regularization technique that randomly sets a fraction of input units to 0 during training,
        # which helps prevent overfitting
        self.dropout = nn.Dropout(p=drop_out)

    # Code to define the forward method of the Classifier class. This method takes in an input tensor x
    # and performs the forward pass through the neural network
    def forward(self, x):

        # To reshape the input tensor x to have a shape of (batch_size, input_size). The -1 in the
        # second dimension means that it will automatically calculate the size based on the other
        # dimensions.
        x = x.view(x.shape[0], -1)

        # Code to iterate over each hidden layer in the hidden_layers list
        for layer in self.hidden_layers:

            # To apply the ReLU activation function to the output of each hidden layer, and then
            # apply dropout to the result. The result is stored in the variable x
            x = self.dropout(F.relu(layer(x)))

        # To apply the log softmax function to the output of the last hidden layer, which produces
        # the final output probabilities for each class. The dim=1 argument specifies that the
        # softmax should be applied along the second dimension (the classes dimension).
        # This is the output, so no dropout code here
        x = F.log_softmax(self.output(x), dim=1)

        # To return the final output tensor x from the forward method
        return x
    
    
    # Replacing the 'classifier' part of the pre-trained model with the newly defined feedforward network
# called 'Classifier' from the previous cell in order to customize the pre-trained model to fit my
# new dataset for training purposes.

input_size = 25088    # input_size is the number of input features to the classifier. In this case, it is set to 25088.
output_size = 102     # output_size is the number of output classes. Here, it is set to 102.
hidden_layers = [4096, 1024]   # hidden_layers is a list that specifies the number of nodes in each hidden layer. In this example, there are two hidden layers with 4096 and 1024 nodes respectively.
drop_out = 0.2      # drop_out is the dropout rate, which is a regularization technique used to prevent overfitting. It randomly sets a fraction of input units to 0 during training. Here, it is set to 0.2, meaning 20% of the input units will be dropped out during training.


# Code to assign a new classifier to the model. 'model' refers to the pre-trained model that I am
# working with, and model.classifier is the classifier part of the model that will be replaced.
# Classifier(input_size, output_size, hidden_layers, drop_out) is a custom classifier class that takes
# in the input size, output size, hidden layers, and dropout rate as arguments. It creates
# fully-connected layers with the specified architecture.
model.classifier = Classifier(input_size, output_size, hidden_layers, drop_out)


# Code to define the loss function. NLL stands for Negative Log Likelihood, and it is commonly used
# for classification problems. This loss function is often used with the LogSoftmax activation
# function in the final layer of the neural network.
criterion = nn.NLLLoss()


# Defining the weights optimizer (backpropagation with gradient descent)
# The optim.Adam() function is used to define the optimizer. In this case, the Adam optimizer is used,
# which is a popular optimization algorithm for training neural networks. It uses adaptive learning
# rates for each parameter, which helps in faster convergence.
# The model.classifier.parameters() specifies that only the parameters of the classifier layer should
# be optimized, while the parameters of the pre-trained feature layers should be frozen. This is a
# common practice in transfer learning, as the lower-level features learned by the pre-trained layers
# are usually more general and transferable to new tasks.
# The learning rate determines the step size at each iteration during the optimization process.
# A smaller learning rate can lead to slower convergence but may result in better optimization,
# while a larger learning rate can lead to faster convergence but may cause overshooting.
# Hence learning rate (lr) is set to 0.003 here.
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# Creating a function called testClassifier used for validation and testing of a classifier model.
# model is the classifier model that will be tested. criterion is the loss function used to calculate
# the loss. testloader is the data loader that contains the test dataset. current_device is the device
# (GPU or CPU) on which the model and data will be moved. This function can be used to evaluate the
# performance of a classifier model on a test dataset. The test loss provides an indication of how
# well the model is performing, while the accuracy represents the percentage of correctly predicted labels.

def testClassifier(model, criterion, testloader, current_device):

    # The model is moved to the current device using the to() method. This ensures that the model and
    # data are on the same device for computation.
    model.to(current_device)

    # Two variables, test_loss and accuracy, are initialized to 0. These variables will be used to
    # calculate the overall loss and accuracy of the model.
    test_loss = 0
    accuracy = 0

    # Code to loop through the images in the testloader data loader. In each iteration, a batch of
    # images and their corresponding labels are retrieved.
    for inputs, labels in testloader:

        # Moving input and label tensors to the default device
        inputs, labels = inputs.to(current_device), labels.to(current_device)

        # The forward pass is performed on the model using the forward() method. This generates the
        # predicted probabilities for each class.
        log_ps = model.forward(inputs)

        # The batch loss is calculated by comparing the predicted probabilities with the actual labels
        # using the criterion loss function.
        batch_loss = criterion(log_ps, labels)

        # The batch loss is added to the test_loss variable.
        test_loss += batch_loss.item()

        # The predicted probabilities are converted to a softmax distribution using the torch.exp()
        # function.
        ps = torch.exp(log_ps)

        # Comparing the highest probability predicted class with the labels. The highest probability
        # and corresponding class index are determined using the topk() method.
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)

        # The predicted class indices are compared with the actual labels to calculate the number of
        # correct predictions. The accuracy is calculated by taking the mean of the correct
        # predictions and adding it to the accuracy variable.
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    # After looping through all the batches, the function returns the total test loss and accuracy.
    return test_loss, accuracy


# Defining a function called trainClassifier used for training a model. The function takes in several
# parameters:
# model: The neural network model to be trained.
# epochs_no: The number of epochs (passes through the dataset) for training.
# criterion: The loss function used to calculate the loss between the predicted and actual labels.
# optimizer: The optimization algorithm used to update the model's weights.
# trainloader: The data loader for the training dataset.
# validloader: The data loader for the validation dataset.
# current_device: The device (GPU or CPU) on which the model and data should be moved.

def trainClassifier(model, epochs_no, criterion, optimizer, trainloader, validloader, current_device):

    # To move the model and data to current hardware config (GPU or CPU)
    model.to(current_device)

    # Initializing (setting the initial values of) several variables
    epochs = epochs_no
    steps = 0
    print_every = 1
    running_loss = 0

    # The training loop begins, iterating over each epoch.
    for epoch in range(epochs):

        # Within each epoch, the model is set to training mode using the train() method.
        model.train()

        # Another loop iterates over the batches of images and labels in the training dataset.
        for inputs, labels in trainloader:

            steps += 1

            # The inputs and labels are moved to the specified device using the to() method.
            inputs, labels = inputs.to(current_device), labels.to(current_device)

            # The gradients of the optimizer are cleared using the zero_grad() method so they do not
            # accumulate.
            optimizer.zero_grad()

            # The forward pass is performed by passing the inputs through the model, and the predicted
            # outputs are obtained.
            log_ps = model(inputs)

            # The loss between the predicted outputs and the actual labels is calculated using the
            # specified loss function.
            loss = criterion(log_ps, labels)

            # The gradients are computed using the backward() method.
            loss.backward()

            # The optimizer updates the model's weights using the step() method.
            optimizer.step()

            # The running loss is updated by adding the current loss value.
            running_loss += loss.item()

        # After a certain number of steps (determined by when 'steps' modulus 'print_every' equals zero),
        # the model is switched to evaluation mode using the eval() method.
        if steps % print_every == 0:

            model.eval()

            # Gradients are turned off for validation to save memory and computations
            # using the torch.no_grad() context manager.
            with torch.no_grad():

                # The testClassifier function is called to validate the model on the validation
                # dataset, and the test loss and accuracy are obtained.
                test_loss, accuracy = testClassifier(model, criterion, validloader, current_device)

            # The average train loss, test loss, and test accuracy are calculated.
            train_loss = running_loss/print_every
            valid_loss = test_loss/len(validloader)
            valid_accuracy = accuracy/len(validloader)

            # The metrics are printed.
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss:.3f}.. "
                  f"Test loss: {valid_loss:.3f}.. "
                  f"Test accuracy: {valid_accuracy:.3f}")

            # The running loss is reset to 0.
            running_loss = 0

            # The model is switched back to training mode using the train() method.
            model.train()

    # Return last metrics. After all epochs are completed, the function returns the last calculated
    # train loss, test loss, and test accuracy.
    return train_loss, valid_loss, valid_accuracy


# Training session 1 - first run - the baseline
# Setting some hyperparameters for the training process. In this case, the dropout rate is set to 0.2,
# which means that during training, 20% of the neurons in the classifier layers will be randomly
# "dropped out" or deactivated to prevent overfitting. The learning rate is set to 0.003, which
# determines the step size at which the optimizer adjusts the weights during training.

drop_out = 0.2
learning_rate = 0.001

# Defining an optimizer using the Adam optimization algorithm. The optimizer is responsible
# for updating the weights of the classifier layers during training. In this case, the optimizer is
# applied only to the parameters of the classifier, while the parameters of the pre-trained feature
# layers are frozen and not updated.
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Specifying the number of epochs, which is the number of times the entire training dataset will be
# passed through the network during training. In this case, the number of epochs is set to 5.
epochs_no = 2


# Calling the trainClassifier() function to train and validate the neural network classifier.
# This function takes as input the model, the number of epochs, the criterion (loss function),
# the optimizer, the training and validation data loaders, and the current device (CPU or GPU).
train_loss, valid_loss, valid_accuracy = trainClassifier(model, epochs_no, criterion, optimizer, trainloader, validloader, current_device)


# After training and validation, a final summary is printed, including the train loss, test loss,
# and test accuracy.
print("Final result \n",
      f"Train loss: {train_loss:.3f}.. \n",
      f"Test loss: {valid_loss:.3f}.. \n",
      f"Test accuracy: {valid_accuracy:.3f}")


# Code to save training progress and call the saveCheckpoint
def saveCheckpoint(model):
    filename = 'model.pt'
    torch.save(model, filename)
    return filename


# Training session 2 (aiming at a test accuracy of at least 90%). This time, we use a learning rate of 0.001, a reduction by a factor of 10.
# to see if accuracy continues improving

drop_out = 0.2
learning_rate = 0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
epochs_no = 5

train_loss, valid_loss, valid_accuracy = trainClassifier(model, epochs_no, criterion, optimizer, trainloader, validloader, current_device)

print("Final result \n",
      f"Train loss: {train_loss:.3f}.. \n",
      f"Test loss: {valid_loss:.3f}.. \n",
      f"Test accuracy: {valid_accuracy:.3f}")


# Code to save training progress and call the saveCheckpoint
filename = 'model.pt'
torch.save(model, filename)
print(filename)


# TODO: Do validation on the test set

# The code snippet below sets the model into evaluation mode, performs testing on the test dataset,
# and prints the test accuracy.

# Setting the model into evaluation mode. In PyTorch, when a model is in evaluation mode, it disables
# certain operations like dropout and batch normalization, which are typically used during training
# but not during testing or inference. This ensures that the model behaves consistently during testing.
model.eval()

# Context manager provided by PyTorch that temporarily disables gradient calculation. During testing
# or inference, we don't need to calculate gradients because we are not updating the model's
# parameters. Disabling gradient calculation helps to save memory and computation.
with torch.no_grad():

    # Calling the testClassifier function, passing in the model, criterion, testloader, and
    # current_device as arguments. This function is responsible for testing the model on the test
    # dataset and returning the test loss and accuracy.
    test_loss, accuracy = testClassifier(model, criterion, testloader, current_device)

# Printing the test accuracy. The accuracy is calculated by dividing the accuracy by the length of the
# testloader, which represents the number of batches in the test dataset. The :.3f format specifier
# ensures that the accuracy is displayed with three decimal places.
print(f"Test accuracy: {accuracy/len(testloader):.3f}")


# TODO: Save the checkpoint

def saveCheckpoint(model):

    '''
    Defines a function called saveCheckpoint that takes a model as input. The purpose of this
    function is to save the model's checkpoint, which includes important information about the model's
    architecture, parameters, and other metadata

    '''

    # Mapping the classes to indices
    model.class_to_idx = train_data.class_to_idx

    # Creating a dictionary called checkpoint. This dictionary contains the following key-value pairs:
    # 'name': The name of the model.
    # 'class_to_idx': The mapping of classes to indices.
    # 'classifier': The model's classifier (assuming it has one).
    # 'model_state_dict': The state dictionary of the model, which includes all the learnable
    # parameters of the model.
    checkpoint = {
        'name': model.name,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict()
    }

    # Generating a unique file name for the checkpoint by appending the current date and time to the
    # string 'model_'. The time.strftime function is used to format the current date and time.
    timestr = time.strftime("%Y%m%d_%H%M%S")
    file_name = 'model_' + timestr + '.pth'

    # The torch.save function is called to save the checkpoint dictionary to a file with the generated
    # file name. The file is saved with the .pth extension, which is commonly used for PyTorch checkpoints.
    torch.save(checkpoint, file_name)

    # The function returns the file name of the saved checkpoint.
    return file_name


# To show the filename
filename