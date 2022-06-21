# Kickstarter Project Status Prediction

An exploration of several Classification and Clustering ML methods to predict the success/failure of a Kickstarter project upon launch. 

Kickstarter is a popular crowdfunding platform where creators can pitch their projects to anyone online and offer rewards (such as discounts and exclusive products) to 'backers' in exchange for funding to bring their project to life.

All codefiles are commented but a brief description of steps followed is as follows:

## Dataset
Information from over 20,000 projects was scraped from Kickstarter and can be found in the file 'Kickstarter.xlsx' within the repo.

## Steps followed
1. Defining the target variable (whether a project was successful or not)
2. Correlation check for independent variables
3. Selection of variables to use in model-building.
    - Important: Since prediction is to be made during project launch, several variables (e.g. amount backed) for which a value can only be derived DURING a project's active period had to be dropped.
4. 1-hot encoding for categorical predictors
5. Train-test split for dataset
6. Build models and perform cross-validation (more details below)
7. Observe accuracy of models and repeat steps 3-6, tuning hyperparameters & trying different combinations of predictors, in an iterative process to achieve a high and robust accuracy value.

## Models used
Several models were built and compared in this project. These are as follows:
- Random Forest
- Gradient Boosting
- Adaboost
- Neural Network (Multilayer Perceptron or MLP)

<br>

The code contained in this repository is used within the course 'INSY 662 Data Mining and Visualization' of the MMA Program at McGill University.