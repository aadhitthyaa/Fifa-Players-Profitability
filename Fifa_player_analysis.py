# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:54:05 2020

@author: Aadhitthyaa H
"""

#import libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
import os

#import sklearn libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report

sns.set(style = "white", context="notebook", palette="deep")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


from google.colab import files

uploaded = files.upload()

#Read File
FifaData = pd.read_csv('Final_Fifa_Data.csv')
#View Dataframe and rename the response variable column
FifaData.head()


# y - Gold_Overall, x - Accelerations, Reactions, Positioning, GK reflexes, Free kick accuracy, Ball control, Composure, Balance
#FifaDataSubset = FifaData[['GOLD_Overall', 'Acceleration', 'Reactions', 'Positioning', 'GK reflexes', 'Free kick accuracy', 'Ball control', 'Balance']]
ColumnsSubset = ['GOLD_Overall',  'Acceleration', 'Reactions', 'Positioning', 'GK reflexes', 'Free kick accuracy', 'Ball control', 'Composure']
FifaDataSubset = FifaData[ColumnsSubset]
FifaDataSubset.head()

#Summary Fifa Data
FifaDataSubset.describe()

#Plot correlations among these variables
sns.heatmap(FifaDataSubset.corr(), annot = True)

FifaDataSubset.dtypes

sns.pairplot(FifaDataSubset[ColumnsSubset])

#Plot all x variable distributions to be categorized by Gold/No Gold to find out the characteristics of a Gold player
for col in FifaDataSubset.iloc[:,1:].columns.tolist():
    plt.hist(FifaDataSubset[FifaDataSubset['GOLD_Overall'] == 1][col], bins = 20, label = "Gold", fc = (1,0,0, 1))
    plt.hist(FifaDataSubset[FifaDataSubset['GOLD_Overall'] == 0][col], bins = 20, label = 'Not Gold', fc = (0,1,0, 0.3) )
    plt.legend()
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
    
#Machine Learning modeling
train = FifaDataSubset[:1310]
X_train = train.drop('GOLD_Overall', axis = 1)
test = FifaDataSubset[1310:2619]
X_test = test.drop('GOLD_Overall', axis = 1)

Y_train = train['GOLD_Overall']
Y_test = test['GOLD_Overall']

#Cross validation
kfold = StratifiedKFold(n_splits = 10)
random_state = 5
classifiers = []
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(DecisionTreeClassifier(random_state= random_state))
classifiers.append(RandomForestClassifier(random_state = random_state))
classifiers.append(GradientBoostingClassifier(random_state= random_state))
classifiers.append(AdaBoostClassifier(random_state = random_state))
classifiers.append(MLPClassifier(random_state = random_state))

#Only Logistic Regression Model
for model in classifiers:
  Model = model
  Model = Model.fit(X_train, Y_train)
  ModelPredictTrain = Model.predict(X_train)
  ModelPredictTest = Model.predict(X_test)
  display(model)
  
  display(accuracy_score(y_true = Y_train, y_pred = ModelPredictTrain))
  display(accuracy_score(y_true = Y_test, y_pred = ModelPredictTest))
  display ("R2 scores")
  display(r2_score(y_true = Y_train, y_pred = ModelPredictTrain))
  display(r2_score(y_true = Y_test, y_pred = ModelPredictTest))
  display("------")
  
#Store CV scores
CV_Results = []
for classifier in classifiers:
   CV_Results.append(cross_val_score(classifier, X_train, Y_train, scoring = "accuracy", cv = kfold, n_jobs = 4))

CV_means = []
CV_std = []

#Store CV means and std
for cv_result in CV_Results:
    CV_means.append(cv_result.mean())
    CV_std.append(cv_result.std())

sns.set_style("ticks", {"xtick.major.size":8,"ytick.major.size":8})
#Store Classifier algorithms in a list
ClassifierAlgos = ['Logistic Regression', 'kNN', 'Decision Tree','Random Forest', 'Gradient Boosting', 'AdaBoost', 'Neural Network']
fig = sns.barplot(y=ClassifierAlgos, x = CV_means, orient = "h")
#fig.set_xticklabels(fig.get_xticklabels(), rotation = 90)
fig.set_xlabel('Accuracy')
fig.set_ylabel('Model')
plt.title('Model Accuracy Comparison')

#Ensemble voting classifier
from sklearn.ensemble import VotingClassifier
VotingClassifier = VotingClassifier(estimators = [('kNN', classifiers[1]), ('Logistic Regression', classifiers[0]), ('GradientBoosting', classifiers[4]), ('AdaBoost', classifiers[5])], voting = 'soft')
EnsembleModel = VotingClassifier.fit(X_train, Y_train)

display(X_test.head())
display(X_train.head())
display(Y_train.head())

#Training prediction
Gold_Prediction_Ensemble_Train = pd.Series(EnsembleModel.predict(X_train), name = "Gold_Player_Predict_Train")
display(confusion_matrix(y_true = Y_train, y_pred = Gold_Prediction_Ensemble_Train))
display(accuracy_score(y_true = Y_train, y_pred = Gold_Prediction_Ensemble_Train))
display(r2_score(y_true = Y_train, y_pred = Gold_Prediction_Ensemble_Train))

Predicted_Prob = []
for i in range(0,len(EnsembleModel.predict_proba(X_test))):
  Predicted_Prob.append(EnsembleModel.predict_proba(X_test)[i][1])
  
Predicted_Prob_Train = []
for i in range(0,len(EnsembleModel.predict_proba(X_train))):
  Predicted_Prob_Train.append(EnsembleModel.predict_proba(X_train)[i][1])
  
#Testing Prediction
Gold_Prediction_Ensemble_Test = pd.Series(EnsembleModel.predict(X_test), name = "Gold_Player_Predict_Test")
display(confusion_matrix(y_true = Y_test, y_pred = Gold_Prediction_Ensemble_Test))
display(accuracy_score(y_true = Y_test, y_pred = Gold_Prediction_Ensemble_Test))
display(r2_score(y_true = Y_test, y_pred = Gold_Prediction_Ensemble_Test))

#Training
fpr, tpr, thresholds = roc_curve(y_true = Y_train, y_score = Predicted_Prob_Train)
roc_auc = auc(fpr, tpr)

#Training
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#Testing
fpr, tpr, thresholds = roc_curve(y_true = Y_test, y_score = Predicted_Prob)
roc_auc = auc(fpr, tpr)

#Testing
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

Ensemble_Model_Predict = pd.concat([Gold_Prediction_Ensemble_Train, Gold_Prediction_Ensemble_Test]).reset_index().drop(['index'], axis = 1).rename(columns = {0: 'Ensemble_Model_Predict'})

Ensemble_Model_Predict.index.is_unique

#FifaDataSubset2 = FifaDataSubset.drop_duplicates(inplace = True)
FifaDataNew = pd.concat([FifaData, Ensemble_Model_Predict], axis = 1)

FifaDataNew.to_csv('FifaDataPredict.csv')
files.download('FifaDataPredict.csv')