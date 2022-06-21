from datetime import date
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.classification import precision_score
from sklearn.metrics.scorer import SCORERS
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

df_import = pd.read_excel("Kickstarter.xlsx")

################## Data Preprocessing & Notes ##################

# Drop occurrences with irrelevant target states
df_filtered = df_import[ (df_import.state == "failed") | (df_import.state == "successful") ]

# (TEST) Fill NaN in category
holes = {"category": "None"}
df_filtered = df_filtered.fillna(value=holes)

# Check for correlated predictors
correlations = df_filtered.corr(method='pearson')
#correlations.to_csv("indiv_proj_corr2.csv")

# Usd_pledged | pledged - 0.9539339
# backers_count | pledged - 0.72921844
# USD_pledged | backers_count - 0.7602639
# Staff_picked | spotlight - 0.34722496

# Check for unary variables
with open("unary_test.txt", mode = 'w', encoding= "utf8") as filewriter:
    for col in df_filtered:
        values = df_filtered[col].value_counts()
        filewriter.write(f"\nOccurrences of each unique value in column {col} :\n {values}")
        filewriter.write("\n\n")
    filewriter.close()
## disable_communication is unary

## Exclude variables:
# project_id/name - irrelevant, too specific
# pledged - correlated with usd_pledged, a better option due to standard measure
# disable_communication : unary
# currency - irrelevant, same reason as pledged
# all deadline vars - launch_to_deadline_days is a better predictor 
# all state_changed vars - seems irrelevant as project's status can be changed at anytime by owners or kickstarter for whatever reason
# all created_at - create_to_launch_days is a better predictor
# all launched_at - create_to_launch_days is a better predictor
# static_usd_rate - irrelevant, same reason as pledged
# spotlight - only TRUE when project is successful. only FALSE when project is failed. Direct correlation with target. 
# name_len/name_len_cleaned - both correlated, doens't seem meaningful to keep
# blurb_len/blurb_len_cleaned - both correlated, latter MIGHT be useful to keep

## Worth noting
# staff_pick - when true, projects higher chance to succeed. when false, projects mixed chance at success. might be good variable
# category - 1471 null values exist. but definitely worth keeping as a variable. replace with none for NaN values.
# blurb_len_cleaned - length of blurb text in project description. Might be good?

################## Split into train/test, and get dummies for categorical ##################

# Convert goal to usd for consistency
df_filtered["usd_goal"] = df_filtered['static_usd_rate'] * df_filtered['goal']


# Separation into in/dependent
y = df_filtered['state']
# X2 = df_filtered[['goal', 'country', 'staff_pick', 'backers_count', 'usd_pledged', 'category', 'name_len_clean', 'blurb_len_clean', 'create_to_launch_days', 'launch_to_deadline_days']]
# X = df_filtered[['goal', 'country', 'staff_pick', 'backers_count', 'usd_pledged', 'category', 'create_to_launch_days', 'launch_to_deadline_days']]
# X3 = df_filtered[['goal', 'country', 'staff_pick', 'backers_count', 'usd_pledged', 'name_len_clean', 'blurb_len_clean','create_to_launch_days', 'launch_to_deadline_days']]
# X4 = df_filtered[['goal', 'country', 'staff_pick', 'backers_count', 'usd_pledged', 'create_to_launch_days', 'launch_to_deadline_days']]
# XC = df_filtered[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','usd_goal']]
# XC2 = df_filtered[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','name_len_clean', 'blurb_len_clean','usd_goal']]
# XC3 = df_filtered[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','deadline_weekday', 'name_len_clean', 'blurb_len_clean','usd_goal']]
# XC4 = df_filtered[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','deadline_weekday', 'usd_goal']]
# XC5 = df_filtered[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','deadline_weekday', 'created_at_weekday', 'launched_at_weekday','usd_goal']]
XC6 = df_filtered[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','deadline_weekday', 'created_at_weekday', 'launched_at_weekday','name_len_clean', 'blurb_len_clean','usd_goal']]
# XCX = df_filtered[['country','category','launch_to_deadline_days','create_to_launch_days','usd_goal']]

## To consider: created_at_weekday, launched_at_weekday,deadline_weekday, name_len_clean, blurb_len_clean

# Encode Categorical Data
X = pd.get_dummies(XC6, columns = ['country', 'category', 'deadline_weekday', 'created_at_weekday','launched_at_weekday'])


################## Feature Selection ##################
'''
##### Find importance through Random Forest
importantforest = RandomForestClassifier(random_state = 13, n_estimators=1000, bootstrap=True, oob_score=True, max_features='sqrt')
model = importantforest.fit(X,y)
print(model.feature_importances_)
coeff_list_rf = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','feature importance']).sort_values(by='feature importance', ascending=False)

##### RFE (standardization not necessary)
lr = LogisticRegression(max_iter=5000)
rfe = RFE(lr, n_features_to_select = 1)
model = rfe.fit(X, y)
model.ranking_
coeff_list_rfe = pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','coefficient']).sort_values(by='coefficient')

# Write results to file
with open("feature_importance_clustering.txt", mode = 'w', encoding= "utf8") as filewriter:
    for index,row in coeff_list_rf.iterrows():
        filewriter.write(f"{row['predictor']} \t {row['feature importance']}")
        filewriter.write("\n")

    filewriter.write("\n\n\n")
    
    for index,row in coeff_list_rfe.iterrows():
        filewriter.write(f"{row['predictor']} \t {row['coefficient']}")
        filewriter.write("\n")
    filewriter.close()
'''

##### Drop insignificant predictors
# country_LU, country_SG, country_HK, country_NL, country_AT, country_DK, country_MX, country_NO, country_NZ, country_ES
# category_Academic, category_Robots, category_Comedy, category_Thrillers, category_Blues, category_Webseries, launched_at_weekday_Saturday, launched_at_weekday_Sunday

### Predictors that only give 1 target state if present
# category_Academic, category_Blues, category_Thrillers, category_Webseries
# category_Places (impt), category_Shorts (impt)

# Drop some predictors
# X = X.drop(['country_LU', 'country_SG', 'country_HK', 'country_NL', 'country_AT', 'country_DK', 'country_MX', 'country_NO', 'country_NZ', 'country_ES'], axis=1)
# X = X.drop(['category_Academic', 'category_Robots', 'category_Comedy', 'category_Robots', 'category_Thrillers', 'launched_at_weekday_Saturday', 'launched_at_weekday_Sunday'], axis=1)


################## Define functions for CV and classifiers ##################

# Default CV object
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=13)

# Standardizes a df with z-score and returns it
from sklearn.preprocessing import StandardScaler
def standardizer(in_df: pd.DataFrame) -> pd.DataFrame:
    standardizer = StandardScaler()
    return standardizer.fit_transform(in_df)

''' 
Creates and runs a RandomForestClassifier with cv to test a provided X and y.
Hyperparams : n_estimators

Params:
    X               - Predictors 
    y               - Targets
    num_trees       - Number of trees to grow. 1300 if unspecified
    param_tuning    - Optional. If provided, runs GridSearchCV on the provided params

Returns: If param_tuning specified, GridSearchCV object. Otherwise, numpy ndarray from cross_val_score. Also returns the model used.
'''
from sklearn.ensemble import RandomForestClassifier
def random_forest(X: pd.DataFrame, y: pd.DataFrame, num_trees: int = 1300, param_tuning: dict = None):
    # Can vary n_estimators
    randomForest = RandomForestClassifier(random_state=13, n_estimators=num_trees, bootstrap=True, oob_score=True, max_features='sqrt')

    if param_tuning == None:
        scores = cross_val_score(randomForest, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores, randomForest
    else:
        grid_search = GridSearchCV(estimator=randomForest, scoring='accuracy', param_grid=param_tuning, cv=cv, n_jobs=-1)
        grid_result = grid_search.fit(X, y)
        return grid_result, grid_search

''' 
Creates and runs gradient boosting with cv to test a provided X and y.
Hyperparams : n_estimators, max_depth (impt)

Params:
    X               - Predictors 
    y               - Targets
    num_trees       - Number of trees to grow. 325 if unspecified
    depth           - Maximum depth of each tree. 3 if unspecified
    param_tuning    - Optional. If provided, runs GridSearchCV on the provided params

Returns: If param_tuning specified, GridSearchCV object. Otherwise, numpy ndarray from cross_val_score. Also returns the model used.
'''
from sklearn.ensemble import GradientBoostingClassifier
def boosted_forest(X: pd.DataFrame, y: pd.DataFrame, num_trees: int = 325, depth: int = 3, param_tuning: dict = None):
    # Can vary n_estimators
    boostedForest = GradientBoostingClassifier(random_state=13, n_estimators=num_trees, max_depth=depth)
 
    if param_tuning == None:
        scores = cross_val_score(boostedForest, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores, boostedForest
    else:
        grid_search = GridSearchCV(estimator=boostedForest, scoring='accuracy', param_grid=param_tuning, cv=cv, n_jobs=-1)
        grid_result = grid_search.fit(X, y)
        return grid_result, grid_search


''' 
Creates and runs an MLP Classifier with cv to test a provided X and y.
Hyperparams : hidden_layer_sizes

Params:
    X               - Predictors 
    y               - Targets
    hidden_layers   - Tuple, neurons and no. of hidden layers. (20) if unspecified
    param_tuning    - Optional. If provided, runs GridSearchCV on the provided params

Returns: If param_tuning specified, GridSearchCV object. Otherwise, numpy ndarray from cross_val_score. Also returns the model used.
'''
from sklearn.neural_network import MLPClassifier
def neural_net(X: pd.DataFrame, y: pd.DataFrame, hidden_layers: tuple = (20), param_tuning: dict = None):
    X = standardizer(X)
    neural = MLPClassifier(random_state=13, hidden_layer_sizes=hidden_layers, activation='relu', solver='adam', max_iter=1000)
 
    if param_tuning == None:
        scores = cross_val_score(neural, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores, neural
    else:
        grid_search = GridSearchCV(estimator=neural, scoring='accuracy', param_grid=param_tuning, cv=cv, n_jobs=-1)
        grid_result = grid_search.fit(X, y)
        return grid_result, grid_search

'''
Creates and runs adaboost with cv to test a provided X and y.
Hyperparams : learning_rate, n_estimators (impt)

Params:
    X               - Predictors 
    y               - Targets
    num_trees       - Number of trees to grow. 250 if unspecified
    learning        - Learning rate. 0.1 if unspecified
    param_tuning    - Optional. If provided, runs GridSearchCV on the provided params

Returns: If param_tuning specified, GridSearchCV object. Otherwise, numpy ndarray from cross_val_score. Also returns the model used.
'''
from sklearn.ensemble import AdaBoostClassifier
def adabooster(X: pd.DataFrame, y: pd.DataFrame, num_trees: int = 250, learning: float=0.1, param_tuning: dict = None):
    ada = AdaBoostClassifier(random_state=13, n_estimators=num_trees, learning_rate=learning)
 
    if param_tuning == None:
        scores = cross_val_score(ada, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores, ada
    else:
        grid_search = GridSearchCV(estimator=ada, scoring='accuracy', param_grid=param_tuning, cv=cv, n_jobs=-1)
        grid_result = grid_search.fit(X, y)
        return grid_result, grid_search


################## Develop Models and Run Tests ##################

##### Simple XCV tests 

# scores_gb, boosted1 = boosted_forest(X=X, y=y) 
# print('Accuracy of boosted forest with 10-fold CV, 4 repeats: %.3f (STD: %.3f)' % (np.mean(scores_gb), np.std(scores_gb)))
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 13) 

# grid_gb = dict()
# grid_gb['n_estimators'] = [275, 300, 325, 330, 350]
# #grid_gb['n_estimators'] = [150, 175, 200, 250, 300, 500]
# grid_gb['max_depth'] = [2, 3, 4, 5]
# scores_gb_grid, boosted2 = boosted_forest(X=x_train, y=y_train, param_tuning=grid_gb)
# print(f"Grid Search - Boosted Forest\nBest score: {scores_gb_grid.best_score_} \t\tUsing: {scores_gb_grid.best_params_}")

# y_pred_gb_grid = boosted2.predict(x_test)
# print(f"Accuracy of tuned boosted forest model (train_test_split): {accuracy_score(y_test, y_pred_gb_grid)}")

scores_rf, forest1 = random_forest(X=X, y=y)
print('\nAccuracy of random forest with 10-fold CV, 4 repeats: %.3f (STD: %.3f)' % (np.mean(scores_rf), np.std(scores_rf)))

scores_gb, boosted1 = boosted_forest(X=X, y=y) 
print('Accuracy of boosted forest with 10-fold CV, 4 repeats: %.3f (STD: %.3f)' % (np.mean(scores_gb), np.std(scores_gb)))

scores_nn, nn1 = neural_net(X=X, y=y)
print('Accuracy of neural net with 10-fold CV, 4 repeats: %.3f (STD: %.3f)' % (np.mean(scores_nn), np.std(scores_nn)))

scores_ab, ada1 = adabooster(X=X, y=y)
print('Accuracy of adaboost with 10-fold CV, 4 repeats: %.3f (STD: %.3f)' % (np.mean(scores_ab), np.std(scores_ab)))


##### Split into train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 13) 

rf_model = forest1.fit(x_train, y_train)
gb_model = boosted1.fit(x_train, y_train)
nn_model = nn1.fit(x_train, y_train)
ab_model = ada1.fit(x_train, y_train)

y_pred_rf = rf_model.predict(x_test)
y_pred_gb = gb_model.predict(x_test)
y_pred_nn = nn_model.predict(x_test)
y_pred_ab = ab_model.predict(x_test)

print(f"\nAccuracy of random forest (train_test_split): {accuracy_score(y_test, y_pred_rf)}")
print(f"Accuracy of boosted forest (train_test_split): {accuracy_score(y_test, y_pred_gb)}")
print(f"Accuracy of neural net (train_test_split): {accuracy_score(y_test, y_pred_nn)}")
print(f"Accuracy of adaboost (train_test_split): {accuracy_score(y_test, y_pred_ab)}")

##### GridSearchCV on train/test split dataset 

grid_rf = dict()
#grid_rf['n_estimators'] = [850, 1000, 1200, 1400, 1500, 1700, 1900]
grid_rf['n_estimators'] = [1200, 1300, 1400, 1450, 1500]
scores_rf_grid, forest2 = random_forest(X=x_train, y=y_train, param_tuning=grid_rf)
print(f"\nGrid Search - Random Forest\nBest score: {scores_rf_grid.best_score_} \t\tUsing: {scores_rf_grid.best_params_}")

grid_gb = dict()
grid_gb['n_estimators'] = [275, 300, 325, 330, 350]
#grid_gb['n_estimators'] = [150, 175, 200, 250, 300, 500]
grid_gb['max_depth'] = [2, 3, 4, 5]
scores_gb_grid, boosted2 = boosted_forest(X=x_train, y=y_train, param_tuning=grid_gb)
print(f"Grid Search - Boosted Forest\nBest score: {scores_gb_grid.best_score_} \t\tUsing: {scores_gb_grid.best_params_}")

grid_nn = dict()
grid_nn['hidden_layer_sizes'] = [(20), (5,5,5), (6,6,6), (5,6,7)]
#grid_nn['hidden_layer_sizes'] = [(20), (18), (23), (15,15), (5,5,5), (7,7,7)]
scores_nn_grid, nn2 = neural_net(X=x_train, y=y_train, param_tuning=grid_nn)
print(f"Grid Search - Neural Net\nBest score: {scores_nn_grid.best_score_} \t\tUsing: {scores_nn_grid.best_params_}")

grid_ab = dict()
grid_ab['n_estimators'] = [100, 150, 200, 300, 400, 500]
grid_ab['learning_rate'] = [0.01, 0.05, 0.1, 0.5, 0.7, 1.0]
scores_ab_grid, ada2 = boosted_forest(X=x_train, y=y_train, param_tuning=grid_ab)
print(f"Grid Search - Adaboost\nBest score: {scores_ab_grid.best_score_} \t\tUsing: {scores_ab_grid.best_params_}")

y_pred_rf_grid = forest2.predict(x_test)
print(f"\nAccuracy of tuned random forest model (train_test_split): {accuracy_score(y_test, y_pred_rf_grid)}")
y_pred_gb_grid = boosted2.predict(x_test)
print(f"Accuracy of tuned boosted forest model (train_test_split): {accuracy_score(y_test, y_pred_gb_grid)}")
y_pred_nn_grid = nn2.predict(x_test)
print(f"Accuracy of tuned neural net model (train_test_split): {accuracy_score(y_test, y_pred_nn_grid)}")
y_pred_ab_grid = ada2.predict(x_test)
print(f"Accuracy of tuned adaboost model (train_test_split): {accuracy_score(y_test, y_pred_ab_grid)}")

'''
##### GridSearchCV on full dataset to find hyperparams ##### 
grid_rf_f = dict()
grid_rf_f['n_estimators'] = [800, 900, 1000, 1150]
scores_rf_grid_f, forest3 = random_forest(X=X, y=y, param_tuning=grid_rf_f)
print(f"\nGrid Search - Random Forest\nBest score: {scores_rf_grid_f.best_score_} \t\tUsing: {scores_rf_grid_f.best_params_}")

grid_gb_f = dict()
grid_gb_f['n_estimators'] = [200, 300, 325, 350, 500, 1000]
grid_gb_f['max_depth'] = [3,4,5]
scores_gb_grid_f, boosted3 = boosted_forest(X=X, y=y, param_tuning=grid_gb_f)
print(f"Grid Search - Boosted Forest\nBest score: {scores_gb_grid_f.best_score_} \t\tUsing: {scores_gb_grid_f.best_params_}")

grid_nn_f = dict()
grid_nn_f['hidden_layer_sizes'] = [(20), (17), (25), (10,10,10), (5,5,5)]
scores_nn_grid_f, nn3 = neural_net(X=X, y=y, param_tuning=grid_nn_f)
print(f"Grid Search - Neural Net\nBest score: {scores_nn_grid_f.best_score_} \t\tUsing: {scores_nn_grid_f.best_params_}")

forest1.fit(X,y)
boosted1.fit(X,y)
nn1.fit(X,y)

################## Test on sample dataset ##################

def preprocess_style1(in_df_X: pd.DataFrame, dummies: List = None) -> pd.DataFrame:
    # (TEST) Fill NaN in category
    holes = {"category": "None"}
    df_filled = in_df_X.fillna(value=holes)

    X = pd.get_dummies(df_filled, columns = dummies)

    return X

# Import Test Dataset
kickstarter_grading_df = pd.read_excel("Kickstarter-Grading-Sample.xlsx")

# Filter only records with desired states
kickstarter_grading_df = kickstarter_grading_df[ (kickstarter_grading_df.state == "failed") | (kickstarter_grading_df.state == "successful") ]

# Transform goal column to standardize with USD
kickstarter_grading_df["usd_goal"] = kickstarter_grading_df['static_usd_rate'] * kickstarter_grading_df['goal']

# Setup X and y
#X_grading = kickstarter_grading_df[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','deadline_weekday', 'created_at_weekday', 'launched_at_weekday','name_len_clean', 'blurb_len_clean','usd_goal']]
#X_grading = kickstarter_grading_df[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','usd_goal']]
X_grading = kickstarter_grading_df[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','name_len_clean', 'blurb_len_clean','usd_goal']]
#X_grading = kickstarter_grading_df[['country','staff_pick','category','launch_to_deadline_days','create_to_launch_days','deadline_weekday', 'name_len_clean', 'blurb_len_clean','usd_goal']]

y_grading = kickstarter_grading_df["state"]

dummyvars = ['country', 'category']
X_grading = preprocess_style1(X_grading, dummyvars)

# Make predictions
y_grading_rf = forest1.predict(X_grading)
y_grading_gb = boosted1.predict(X_grading)
y_grading_nn = nn1.predict(X_grading)

y_grading_rf_grid = forest3.predict(X_grading)
y_grading_gb_grid = boosted3.predict(X_grading)
y_grading_nn_grid = nn3.predict(X_grading)


# Calculate the accuracy score
print(f"\nAccuracy of random forest model (sample dataset): {accuracy_score(y_grading, y_grading_rf)}")
print(f"Accuracy of boosted forest model (sample dataset): {accuracy_score(y_grading, y_grading_gb)}")
print(f"Accuracy of neural net model (sample dataset): {accuracy_score(y_grading, y_grading_nn)}")

print(f"\nAccuracy of tuned random forest model (sample dataset): {accuracy_score(y_grading, y_grading_rf_grid)}")
print(f"Accuracy of tuned boosted forest model (sample dataset): {accuracy_score(y_grading, y_grading_gb_grid)}")
print(f"Accuracy of tuned neural net model (sample dataset): {accuracy_score(y_grading, y_grading_nn_grid)}")
'''
'''
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means2 = grid_result.cv_results_['mean_test_score']
        params2 = grid_result.cv_results_['params']
        x3 = [params2[i]['n_estimators'] for i in range(len(params2))]
        z3 = [params2[i]['max_depth'] for i in range(len(params2))]
'''
'''

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search2 = GridSearchCV(estimator=model3, param_grid=grid2, n_jobs=-1, cv=cv, scoring='accuracy')
# execute the grid search
grid_result2 = grid_search2.fit(x, y)


grid2 = dict()
grid2['n_estimators'] = [10, 20, 50,70, 100, 150,200,300,400,500]
grid2['max_depth'] = [3,4,5,6,7,8]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

'''

'''
from sklearn.model_selection import GridSearchCV
dict_params = { 'hidden_layer_sizes' : [i for i in range(1, 22)] }

# Test random states 1-50
for i in range (1, 51):
    model = MLPClassifier(max_iter=1000, random_state=i)
    grid = GridSearchCV(estimator=model, param_grid=dict_params, cv = 5, refit = True, verbose=0)

    # Fit data to model
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test) 

    # Add best hidden_layer size + classification results to dict
    results_random_state[i] = [grid.best_params_, classification_report(y_test, y_pred)]
'''
### References ###
# https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas

'''
with open("unary_test.txt", mode = 'w', encoding= "utf8") as filewriter:
    for col in df_filtered:
        values = df_filtered[col].value_counts()
        filewriter.write(f"\nOccurrences of each unique value in column {col} :\n {values}")
        filewriter.write("\n\n")
    filewriter.close()
# disable_communication is unary

# Drop rows with null values
df_nonull = df.dropna(axis=0, how='any')

'''