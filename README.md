# Fifa-Players-Profitability

Description
In this project, our group aspires to create an effective model that identifies the most profitable FIFA players, so that clubs can acquire high-performing players at a fair cost and enjoy a great value from them in return. In this way, we can help clubs dramatically increase their profits and give them a competitive advantage in future. Our approach is to first thoroughly understand the business problem, and then design a process flow to solve the problem. We began by acquiring in-depth external data from credible online sources such as Kaggle. Once we had all the relevant data, we then cleaned the data by taking out data points with any error and null values, as well as entries that do not carry important business meaning in regards to our objective. After cleaning we merged a data set to bring down the club numbers from 740 to top 100.This helped us in performing profit-loss analysis per player .Then, we created a Logistic Regression model and an Ensemble model to predict the most profitable players more accurately. In the end, we achieved great success from both business and statistical perspective. We are able to help the clubs acquire high-value players at a low cost. Therefore, we can greatly reduce the clubs’ loss or increase their profit. Additionally, we can help the clubs form strong teams that have a high chance of winning future games. In terms of the model success, the Key Results section includes all relevant model performance.

Key Results
We selected a total of seven variables: Acceleration, Reactions, Positioning, GK reflexes, Free kick accuracy, Ball control, Composure, which were selected based on model significance and business knowledge. We first built a logistic regression model with R square of 57.66% for training and 63.29% for testing. At the same time, the accuracy for training is 90.45% and for testing is 91.38%. Then we created our best model, which was an ensemble model of Logistic Regression, kNN, Gradient Boosting and AdaBoost. We achieved 94.6% accuracy on the Training set and 92.5% on the Testing set. The Training R2 was 69% and the Testing R2 of 60%. By applying our best model, we can help the clubs increase their profits by €0.57 million per player.

Tools used

Python
SAS JMP
Excel
Tableau
Methods used
Data Collection
Data Cleaning
ETL
Exploratory Data Analysis
Data Preprocessing
Feature Selection
Feature Extraction
Machine Learning
Data Visualization
Profit/Loss Analysis
