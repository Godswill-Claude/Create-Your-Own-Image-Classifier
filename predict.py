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


# TODO: Write a function that loads a checkpoint and rebuilds the model

def rebuildModel(filepath):
    '''
    The purpose of this function is to rebuild a pretrained model using the saved checkpoint file. It
    takes a file path as input.

    '''

    try:
        # To load the checkpoint file using the torch.load function. The map_location argument is used to
        # specify the device where the model should be loaded. In this case, lambda storage, loc: storage
        # is used to load the model on the CPU.
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

        # To recreate the pretrained base model using the getattr function. The getattr function is used
        # to dynamically access a module or attribute based on its name. In this case, it accesses the
        # model specified by checkpoint['name'] from the models module and initializes it with pretrained
        # weights.
        model = getattr(models, checkpoint['name'])(pretrained=True)

        # Replacing the classifier part of the model with the one saved in the checkpoint file. The
        # checkpoint['classifier'] contains the classifier information.
        model.classifier = checkpoint['classifier']

        # loading the saved state dictionary of the model using the load_state_dict function. The state
        # dictionary contains the parameter values of the model.
        model.load_state_dict(checkpoint['model_state_dict'])

        # To assign the class_to_idx dictionary from the checkpoint file to the class_to_idx attribute of
        # the model. This dictionary maps class labels to their corresponding indices.
        model.class_to_idx = checkpoint['class_to_idx']

        # Move the model to GPU if available
        if torch.cuda.is_available():
            model.to('cuda')

        # To return the rebuilt model.
        return model

    except Exception as e:
        print("Error occurred while rebuilding the model:", str(e))
        return None

    
    
    def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''

    # Check if the width of the image is greater than its height
    if image.width > image.height:
        image.thumbnail((10000000, 256))   # Constraining the height to be 256
    else:
        image.thumbnail((256, 10000000))   # Constraining the width to be 256

    # Perform a center crop on the image
    crop_size = 224
    left_margin = (image.width - crop_size) / 2
    bottom_margin = (image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Convert the image to a NumPy array
    np_image = np.array(image)

    # Normalize the image
    np_image = np_image / 255

    # Standardize the image
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds

    # Transpose the color channels
    np_image = np_image.transpose(2, 0, 1)

    # Return the preprocessed image
    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes it is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


img_ready = Image.open(image_path)
img_ready = process_image(img_ready)

plt.imshow(img_ready.transpose((1, 2, 0)))
plt.show()


model = torch.load('model.pt')


def predict(image_path, model, topk=5):
    ''' Predicts the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    # To move the model into evaluation mode and to CPU
    model.eval()

    # To open the image located at image_path using the PIL library's Image.open() function.
    image = Image.open(image_path)

    # To process the image using a function called process_image(). This function applies
    # transformations to the image, such as resizing, cropping, and normalizing, to prepare it for
    # input into the model.
    image = process_image(image)

    # To convert the processed image, which is a NumPy array, into a PyTorch tensor of type
    # torch.FloatTensor
    image = torch.tensor(image, dtype=torch.float)

    # To add a batch dimension to the image tensor. The model expects input in the shape of
    # (batch_size, channels, height, width), so the unsqueeze() function adds a dimension of size 1 at
    # the beginning.
    image = image.unsqueeze(0)

    # To pass the image tensor through the model's forward method to obtain the predicted
    # probabilities for each class. The torch.exp() function is used to reverse the log conversion
    # that might have been applied to the output probabilities.
    probs = torch.exp(model.forward(image))

    # To select the top topk probabilities and their corresponding labels from the predicted
    # probabilities. The topk() function returns the topk largest elements and their indices.
    top_probs, top_labs = probs.topk(topk)

    # To convert the top probabilities from PyTorch tensors to NumPy arrays and then to a Python list.
    top_probs = top_probs.squeeze().tolist()

    # To create a dictionary that maps the indices of the model's classes to their respective labels.
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}

    # initializes an empty list to store the labels.
    labs = []

    # To convert the top labels to their respective class names, using a list comprehension
    labs = [idx_to_class[label.item()] for label in top_labs.squeeze()]

    # returns the top probabilities and labels as a tuple
    return top_probs, labs


model.class_to_idx = image_datasets['train'].class_to_idx


# Using the predict function to predict the class of an image using our pre-trained model.

# calls the predict function with three arguments: image_path, model_from_file, and topk=5.
# The image_path is the path to the image file, model_from_file is the pre-trained model loaded from
# a file, and topk=5 specifies that the function should return the top 5 predicted probabilities and
# their corresponding classes.
probs, classes = predict(image_path, model, topk=5)

# to see the path of the image being classified, the predicted probabilities for each class, and the
# corresponding class labels.
print(image_path)
print(probs)
print(classes)


def convertCategoryToName(categories, mapper='cat_to_name.json'):

    ''' The purpose of this function is to map the predicted categories (represented as numbers) to
    their corresponding flower names. It does this by loading a JSON file (cat_to_name.json) that
    contains a mapping between category numbers and flower names. The function then retrieves the
    flower names for the predicted categories and returns them as a list.

    '''

    # opens the JSON file specified by the mapper parameter in read mode. The mapper parameter is set
    # to 'cat_to_name.json' by default. The with statement ensures that the file is properly closed
    # after it is used.
    with open(mapper, 'r') as f:

        # loads the contents of the JSON file into the cat_to_name variable. The json.load() function
        # is used to parse the JSON data from the file.
        cat_to_name = json.load(f)

        # initializes an empty list called names. This list will store the flower names corresponding
        # to the predicted categories.
        names = []

        # starts a loop that iterates over each element in the categories list.
        for category in categories:

            # retrieves the flower name corresponding to the current category from the cat_to_name
            # dictionary and appends it to the names list. The str(category) is used to ensure that
            # the category is converted to a string before accessing the dictionary.
            names.append(cat_to_name[str(category)])

    # returns the names list, which contains the flower names corresponding to the predicted categories.
    return names


# To make the predictions more interpretable and easier to understand, we translate the class
# identifiers into names

# calls the convertCategoryToName function with the classes variable as an argument. The classes
# variable contains the predicted class identifiers. The function maps these class identifiers to
# their corresponding names and returns a list of human-friendly names, which is assigned to the
# names variable.
names = convertCategoryToName(classes)

print(probs)
print(classes)
print(names)


def select_random_image(path):

    '''
    The purpose of this function is to randomly select an image file path from a specified parent path.
    It does this by randomly selecting a folder from the parent path, and then randomly selecting a
    file from that folder. The function then returns the full path to the randomly selected image file.
    It takes a single parameter path.

    '''

    # selects a random folder from the list of folders in the specified path directory.
    # The os.listdir() function is used to get a list of all the files and folders in the specified
    # directory, and random.choice() is used to select a random folder from that list.
    random_folder = random.choice(os.listdir(path))

    # selects a random file from the list of files in the randomly chosen folder.
    # The os.path.join() function is used to join the path and random_folder together to create the
    # full path to the folder. Then, os.listdir() is used to get a list of all the files in that
    # folder, and random.choice() is used to select a random file from that list.
    random_file = random.choice(os.listdir(os.path.join(path, random_folder)))

    # creates the full image path by joining the path, random_folder, and random_file together using
    # the os.path.join() function.
    image_path = os.path.join(path, random_folder, random_file)

    # returns the image_path, which is the randomly selected image file path.
    return image_path


def display_preds(path, model, topk=5, flower_names=None):

    '''
    The purpose of this function is to select a random image, predict its class using the provided
    model, and visualize the prediction result. It displays the image along with the predicted class
    probabilities in the form of a bar plot. It takes four parameters: path, model, topk, and
    flower_names.

    '''

    # calls the select_random_image function to get a random image file path from the specified path
    # directory. The image_path variable stores the randomly selected image file path.
    image_path = select_random_image(path)
    #print(image_path)

    # extracts the folder number from the image_path. The image_path is split using the forward slash
    # ('/') as the delimiter, and the third element of the resulting list is assigned to the
    # folder_number variable. This folder number corresponds to the class identifier of the image.
    folder_number = image_path.split('/')[2]
    #print(folder_number)

    # retrieves the flower name based on the folder_number from the flower_names dictionary. The
    # flower_names dictionary maps the class identifiers to their corresponding flower names. The
    # retrieved flower name is assigned to the title variable.
    title = flower_names[folder_number]
    #print(title)

    # calls the predict function with the image_path, model, and topk parameters to predict the class
    # probabilities and class identifiers for the image. The predicted probabilities are assigned to
    # the probs variable, and the predicted class identifiers are assigned to the classes variable.
    probs, classes = predict(image_path, model, topk)
    #print(probs)
    #print(classes)

    # calls the convertCategoryToName function with the classes variable as an argument to convert the
    # class identifiers into their corresponding names. The resulting names are assigned to the names
    # variable.
    names = convertCategoryToName(classes)
    print('n:', names)
    print('c:', classes)

    # opens the image using the Image.open() function from the PIL library. The image variable stores
    # the opened image.
    image = Image.open(image_path)

    # preprocesses the image using the process_image function. The process_image function applies
    # transformations to the image to make it compatible with the PyTorch model.
    image = process_image(image)

    # sets up a plot with a figure size of 6 inches by 10 inches.
    plt.figure(figsize = (6, 10))

    # creates a subplot with 2 rows, 1 column, and selects the first subplot.
    ax = plt.subplot(2, 1, 1)

    # displays the image using the imshow function, passing in the image, ax, and title as
    # arguments. The imshow function is responsible for visualizing the image with the provided title.
    imshow(image, ax, title=title);

    # creates a subplot with 2 rows, 1 column, and selects the second subplot.
    plt.subplot(2, 1, 2)

    # creates a bar plot using the Seaborn library. It visualizes the predicted probabilities (probs)
    # on the x-axis and the corresponding class names (names) on the y-axis. The color of the bars is
    # set using the first color from the default Seaborn color palette.
    sns.barplot(x=probs, y=names, color=sns.color_palette()[0]);

    # displays the plot.
    plt.show()

    # indicates the end of the function and returns nothing.
    return


# TODO: Display an image along with the top 5 classes


# assigns the value 5 to the variable number_of_predictions. This variable determines the number of
# images for which predictions will be performed.
number_of_predictions = 5

# starts a loop that will iterate number_of_predictions times (in this case, 5 times). The loop
# variable i will take on values from 0 to number_of_predictions - 1.
for i in range(number_of_predictions):

    # calls the display_preds function with the arguments 'flowers/test' (the path to the test images
    # directory), model_from_file (the pre-trained model), and flower_names=cat_to_name (the dictionary
    # mapping class identifiers to flower names). This function will select a random image from the
    # test directory, predict its class using the model, and visualize the prediction result.
    # By running this loop, the display_preds function will be called number_of_predictions times,
    # each time selecting a different random image from the test directory and displaying its
    # prediction result.
    display_preds('flowers/test', model, flower_names=cat_to_name)
    
    
    