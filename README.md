# Create-Your-Own-Image-Classifier
## Overview
> In this project, I implemented an image classification application with PyTorch, which built and trained a deep neural network learning model on a data set of images and then used the trained model to classify images. In the first part of the project, I developed my code in a jupyter notebook to make sure my training implementation works. In the second part, I converted the code into an application that others can use. My application was a pair of python scripts ("train.py" and "predict.py") that runs from the command line. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses the trained network to predict the class for an input image. 

## Dataset
> The dataset used for this project is the flower dataset and can be found in the following address: [here]('https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz').

## Programming languages used
> The python programming language, along with libraries such as pandas, numpy, matplotlib and pytorch were used for the explorations and visualizations contained in this work. 

## Summary of Findings
> I succeeded in getting a validation accuracy of about 86% in Test 2 using a learning rate of 0.001 and number of epochs of 5. Hence, these settings were saved as the final model used later for predictions.
> Also, I was able to achieve a test accuracy of 0.836 or 84% on new data the model had never seen, which is not a bad outcome for a model generalizing on new data it had never seen previously. More so, I observed that the difference between validation accuracy and test accuracy (model's score) is not much, which suggests that the model does not have much of an overfitting problem.
