# Background
Q1. Why are bees important?
- A third of the world's food production depends on bees.
- Major pollinators.

Q2. Problem
- Declining bee population.
- Mites and ants infestation.
- Frequent check-ups on the hive is desirable to monitor the hive strength and health.
- Manual investigation of beehives is time intensive and often disrupts the beehive environment.

# Motivation & Objective
How can we help ?
- Improve the ability to understand the hive health without checking for the hive manually.
- Devise a non-invasive hive health checkup technique using ML.
- Studies show that honeybee images flying in/out of the hive can be used to draw inferences about the hive's health.

# Contribution
- Use convolutional neural network (CNN) to design ML models to predict hive health.
- Assess the models for their performances.

# Dataset Description
- Our dataset contains 5100+ bee images annotated with location, date, time, subspecies, health condition, caste, and pollen.
- Dataset link: https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images
- The entire dataset was divided into training, validation, and testing roughly in the ratio of 70:20:10.

# Data Visualization
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234079858-790260a7-4263-47c8-9cf8-7a683fcb8fa6.JPG" width=80% height=50%>
</div>

# Methods & Implementation
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234070947-a8dbec7f-6c51-4146-bee3-5cce4878ebf3.png" width=50% height=30%>
</div>
Based on a given bee (input) image, the classifier would predict the status of the hive. There were six class labels. Convolutional neural network (CNN) has been a popular deep learning model for image classification that has shown remarkable performance. For this project, 2 CNN models are used. 

The first prototype is a CNN written from scratch. There are two convolutional layers followed by a max pooling layer respectively. The first convolutional layer has 32 filters with kernel sizes of 3x3 and a rectified linear unit activation. The second convolutional layer is similar to the first, except it has 64 filters. Each max pooling layer reduces the dimensionality of the previous layer by two. The output layer has six units, which corresponds to the six labels for beehive health. Softmax activation is used to output prediction probability for each label. The model contained 1,223,622 trainable parameters. 

The second CNN is a pretrained model. TensorFlow’s mobile net is chosen as it is a relatively small and efficient CNN2. To make the model suitable (as the model was originally trained for 1,000 different classes) the last six layers were modified so that there were six output classes. In total the model has 3,213,126 trainable parameters.  
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234069628-98f9cf4c-0bf8-4107-9162-c48b10d2645e.png" width=40% height=20%>
</div>

The code was run from Google Colaboratory using GPU mode. For each epoch, the training time varied from 5 to 39 seconds for the first model and from 22 to 24 seconds for the pretrained CNN. Since this was our first attempt towards developing a model using CNN, we referenced a tutorial3 as guidance.

# Code (in Python)
Jupyter Notebook File: https://github.com/ShilpikaB/Beehive-Health-Prediction_CSC869/blob/main/beeimage-classifier.ipynb


# Results
- On training dataset: 99% Accuracy was achieved for both the CNN models.
- On validation dataset: 86% and 88% Accuracy for the initial model and the pretrained models.
- On the test dataset: 86% and 96% Accuracy for the initial and the pretrained models.
- The confusion matrix for these multilabel classifiers and used the matrix to calculate precision, recall, f1 measure, receiver operating characteristic (ROC) curve, and area under curve (AUC) on the test dataset. We use the one-vs-the-rest(OvR) multiclass strategy to derive the ROC and AUC.
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234068926-5a3dfad1-300a-450b-81c5-cb31966e3ea6.JPG" width=80% height=10%>
  <img src="https://user-images.githubusercontent.com/82466266/234068967-cf598347-c252-4604-bf5e-d91cb72445e9.JPG" width=80% height=20%>
  <img src="https://user-images.githubusercontent.com/82466266/234068782-9b4992eb-0837-48b9-be52-d713daa523a8.png" width=80% height=30%>
</div>


# Discussion
In the first model, a unique pattern was observed in the confusion matrix. The vast majority of incorrect predictions for the label ‘HiveBeingRobbed’ belonged to the class label ‘Healthy’ and vice-versa. This indicates that data for these two class labels may have some overlapping attributes as a result of which the model is unable to differentiate the two labels. Looking back into the definitions of a robber bee (which cater to ‘HiveBeingRobbed’ status) and healthy bee, the below assumption can be made.

### Inference: 
Robber bees usually fly towards a hive to destroy the hive and steal any stored nectar. These robber bees have shiny bodies with no pollen. On the other hand, healthy bees when leaving the hive also have shiny bodies and do not have any pollen. They will only have pollen on their bodies when they return to the hive after collecting nectar. Therefore, it is likely for our model to incorrectly distinguish between robber bees and healthy bees since the input data set does not contain any information about the direction with respect to the hive. That is, we cannot conclude whether a bee is flying into or out of the hive which would be essential to differentiate between robber bees flying towards a hive and healthy bees flying away from the hive.

### Conclusion 
- The pretrained CNN model had an improved performance than the CNN model prototype..
- The first model performs poorly for the ‘HiveBeingRobbed’ label. The same is reflected in the ROC and AUC for this label where we see that ‘HiveBeingRobbed’ has the lowest AUC. The pretrained model also did not show any significant improvement for this label. However, the performance for the other class labels were drastically improved. This implies that the pretrained model was able to improve the overall image classification problem and also strengthens our assumption that the issue with ‘HiveBeingRobbed’ is at the input data level and not with the model.

### Future Direction
- Randomize and automate the dataset segregation into the train/validate/test data sets.
- Dataset currently has no way to identify the direction of flight of the bees - towards or away from the hive. The direction of flight of bees helps to identify the difference between the robber and healthy bees.
- MissingQueen data has very few images, and all are most likely for the same hive, therefore the model could be learning based on the image background.


# References
- https://www.kaggle.com/jenny18/honey-bee-annotated-images
- https://arxiv.org/abs/1704.04861
- https://deeplizard.com/learn/video/RznKVRTFkBY
