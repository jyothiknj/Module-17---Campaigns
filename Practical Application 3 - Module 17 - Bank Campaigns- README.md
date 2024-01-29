# Practical Application III: Comparing Classifiers

**Overview**: In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone.  



### Getting Started

Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.  We will make use of the article accompanying the dataset [here](CRISP-DM-BANK.pdf) for more information on the data and features.



### Understanding the Data

To gain a better understanding of the data, read the information provided in the UCI Machine Learning repository link, and examined the Materials and Methods section of the paper.  

Observed this data represents **17 marketing campaigns that occurred between May 2008 and Nov 2010**


Imported all the necessary libraries and read the data from the dataset `bank-additional-full.csv` using pandas. 

1. Observed there aren't any missing values in the dataset, allthough few data points are given a value "unknown". These data values are substituted with meaningful values during Exploratory Data Analysis.

![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/3150d26f-409b-44ad-9d3b-84ef6ef6964e)

2. Looking at the dataset, it's very clear, there are many categorical features like job, education, marital, month, housing etc. These have to be converted into numerical

3. This is an unbalanced dataset as the target variable y, which shows whether the customer subscribed or not, has varied proportions of values. This can be clearly seen using a countplot function from Seaborn.

![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/d0f50e5c-447a-465c-9e24-fc875a5e9ecb)


### Business Objective: 

The business objective is to find a model that can increase the campaign efficiency thus making more clients subscribe. Expectation is to identify the main features that affect success, which can ultimately help with identifying the potential customers

### Exploratory Data Analysis

1. Created countplots for various features to look at the unique values and also the proportions of each value in the data.
Few count plots are as below:

**Job:**
![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/ac7fbea8-c786-49a4-8e05-73fb17cffcf8)

**Marital:**
![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/9f848c8b-0e51-41d2-8bd7-47415d83d446)

**Loan:**
![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/8e2812cb-11d2-408e-9684-13b7f5550d42)

**Housing:**
![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/ec7945d6-23fb-448f-8367-b9447141078c)

**Default:**
![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/b9eeb131-31e0-49cf-b7f7-31470ab5ef3c)

**Education:**
![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/a87ca215-d455-4a04-ab74-9d42b9317e53)

### Feature Engineering:

Based on the recommendation to use just the bank information features (columns 1 - 7), performed the below steps to prepare the features and target column for modeling with appropriate encoding and transformations.

Created a new dataframe,'X' with the recommended features **'age', 'job', 'marital', 'education','default','housing','loan'**
Created a new array,'y' with the target column data 'y'

![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/d2e1bab0-0b83-4499-afff-2057bcc0f475)


1. Converted the target feature 'y' into binary by substituting 0 for 'no' and 1 for 'yes'
2. Updated the 'unknown' values in the 'marital' feature by random assignment from ['married','single','divorced'] values
3. Updated the 'unknown' values in the 'housing','loan' and 'default' features by random assignment from ['yes','no] values. Then converted all the values into binary by substituting 0 for 'no' and 1 for 'yes'
4. Updated the 'unknown' values in the 'marital' feature by random assignment from ['basic.4y','high.school','basic.6y','basic.9y','professional.course','university.degree','illiterate'] values.

5. Now using the get_dummies function, created the dummy columns for categorical features namely, 'job', 'marital' and 'education'. Concatenated these new columns with original dataset and dropped the redundant columns 'job', 'marital' and 'education'

So, the final dataset after all the encodings and transformations, contains 26 columns as below:
![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/1b9427b8-1642-4506-a87c-7f0df5ca203a)


### Train/Test Split

Split the data into Training and Test datasets using train_test_split function with a test_size of 0.3. 

### A Baseline Model

Before we build our first model, established a baseline using DummyClassifier model with strategy 'most_frequent'.  This would be the baseline performance that our classifier will aim to beat.

Fit the training dataset and predicted the target values on the test data.

**Baseline Scores:**

Training dataset accuracy score - 0.887239

Test dataset accuracy score - 0.887594	

### Simple Models with Default Parameter Values

Built a basic model using Logistic Regression. Fit the model using training dataset. The accuracy score is found to be 0.88759407 which is almost same as the baseline.

Now built KNN, Decision Tree and SVM models with the default settings and fit them on the training dataset.
As the dataset is inbalanced, calculated F1 score as well along with looking for Accuracy.

For each model, recorded the training time, training and test accuracy scores, and F1 scores as well. Here are the observations

![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/04f37a1c-c035-4d0b-9381-84dbc98f0b98)

From the observations it's very clear that when run on default parameter values, DecisionTree Classifier has the highest accuracy score on training set and also has the highest F1 scores while SVC model has the highest accuracy score on the test data.But it's almost 8 times slower than the other models. 

So, **when run on default model parameters, DecisionTreeClassifier turned out to be better model in predicting the customers who will subscribe or not after the campaign call.**



### Model Improvisions

We continued to use all the same features ('age', 'job', 'marital', 'education','default','housing','loan') from the dataset and hyperparameter tuning. Used **GridSearchCV** to identify the best parameters for each of the models. This allowed to accurately compare and see the improvement in each model.

Below here are the observations:
![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/ac65cc5e-fca6-4d48-809d-71afe2ad763a)

Plotted the **Confusion Matrix for the KNeighborsClassifier** to look at the number of false positives and false negatives.

From the confusion matrix it's very clear the false positive rate is pretty low. 
This means customers are not falsely predicted as they would subscribe though they don't . So, with the model predictions, campaigns can confidently focus on the customers predicted as negative and work to make them subscribe

![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/a285cf57-19bc-473b-ab7c-0ba95f993056)

Plotted the **ROC (Receiver Operating Curve)** for all the above models. 
![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/9157f981-87f5-4af8-8302-eb9501323530)
      

**Training times:**
Plotted a line plot using seaborn library to visualize the variation in the training times of all the models. From the picture it's evident that SVC takes the longest time to train the data.

![image](https://github.com/jyothiknj/UC-Berkeley-AI-ML-Course-2023-24/assets/35855780/a53034b0-bb8b-485a-9086-1b68207c1b2a)

#### Findings:

This is a very inbalanced dataset. The above observations show clearly that SVC takes longest time to train. Though the test data accuracy time of KNeighborsClassifier is slightly lower than other models, this is relatively faster than others. This can be considered the best model for predicting/classifying the customers that they subscribe or not. 

#### Recommendations:
The data is quite inbalanced. My recommendataion is to go back and collect more datapoints of customers subscribed and add them to the data set. This would reduce the inbalance and we can train the models better and increase their performance.
