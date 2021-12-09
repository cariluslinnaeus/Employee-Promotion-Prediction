#import all the required libraries

import numpy as np

#for dataframe operations
import pandas as pd
#for data visualisation

import seaborn as sns
import matplotlib.pyplot as plt

#for machine learning
import sklearn
import imblearn
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report



#setting up the size of the figures
plt.rcParams['figure.figsize'] = (16,5)
#setting up the style of the plot
plt.style.use('fivethirtyeight')

#reading the datasets
train =pd.read_csv('trainProm.csv')
test = pd.read_csv('testProm.csv')


# lets impute the missing values in the Training Data

train['education'] = train['education'].fillna(train['education'].mode()[0])
train['previous_year_rating'] = train['previous_year_rating'].fillna(train['previous_year_rating'].mode()[0])
train_total = train.isnull().sum()

# lets impute the missing values in the Testing Data

test['education'] = test['education'].fillna(test['education'].mode()[0])
test['previous_year_rating'] = test['previous_year_rating'].fillna(test['previous_year_rating'].mode()[0])




# lets create some extra features from existing features to improve our Model

# creating a Metric of Sum
train['sum_metric'] = train['awards_won?']+train['KPIs_met >80%'] + train['previous_year_rating']
test['sum_metric'] = test['awards_won?']+test['KPIs_met >80%'] + test['previous_year_rating']

# creating a total score column
train['total_score'] = train['avg_training_score'] * train['no_of_trainings']
test['total_score'] = test['avg_training_score'] * test['no_of_trainings']
# lets remove some of the columns which are not very useful for predicting the promotion e.g region, employeeid,recruitment channel 

train = train.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)
test = test.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)
# lets remove the above two columns as they have a huge negative effect on our training data



train = train.drop(train[(train['KPIs_met >80%'] == 0) & (train['previous_year_rating'] == 1.0) & 
      (train['awards_won?'] == 0) & (train['avg_training_score'] < 60) & (train['is_promoted'] == 1)].index)

# lets encode the education in their degree of importance 
train['education'] = train['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))
test['education'] = test['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))

# lets use Label Encoding for Gender and Department to convert them into Numerical


le = LabelEncoder()
train['department'] = le.fit_transform(train['department'])
test['department'] = le.fit_transform(test['department'])
train['gender'] = le.fit_transform(train['gender'])
test['gender'] = le.fit_transform(test['gender'])

# lets split the target data from the train data

y = train['is_promoted']
x = train.drop(['is_promoted'], axis = 1)
x_test = test


#resample the data, as the Target class is Highly imbalanced.
# Here We are going to use Over Sampling Technique to resample the data.using SMOTE
x_resample, y_resample  = SMOTE().fit_resample(x, y.values.ravel())
print("Before Resampling :")
print(y.value_counts())

print("After Resampling :")
y_resample = pd.DataFrame(y_resample)
print(y_resample[0].value_counts())
# create a validation set from the training data so that we can check whether the model that we have created is good enough
# lets import the train_test_split library from sklearn to do that


x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 0)

# It is very import to scale all the features of the dataset into the same scale
# Here, we are going to use the standardization method, which is very commonly used.

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)

# creating prediction model using Decision Trees 
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# lets perform some Real time predictions on top of the Model that we just created using Decision Tree Classifier

# lets check the parameters we have in our Model
'''
department -> The values are from 0 to 8, (Department does not matter a lot for promotion)
education -> The values are from 0 to 3 where Masters-> 3, Btech -> 2, and secondary ed -> 1
gender -> the values are 0 for female, and 1 for male
no_of_trainings -> the values are from 0 to 5
age -> the values are from 20 to 60
previou_year_rating -> The values are from 1 to 5
length_of service -> The values are from 1 to 37
KPIs_met >80% -> 0 for Not Met and 1 for Met
awards_won> -> 0-no, and 1-yes
avg_training_score -> ranges from 40 to 99
sum_metric -> ranges from 1 to 7
total_score -> 40 to 710

predict = model.predict(np.array([[2, #department code
                                      3, #masters degree
                                      0, #male
                                      4, #1 training
                                      60, #30 years old
                                      3, #previous year rating
                                      4, #length of service
                                      1, #KPIs met >80%
                                      3, #awards won
                                      99, #avg training score
                                      7, #sum of metric 
                                      600 #total score
                                     ]]))
print(predict)
'''
def prediction(department,education,gender,no_of_training,age,rating,experience,kpis_met,awards_won,score,sum_metric,total_score):
    return model.predict(np.array([[department,education,gender,no_of_training,age,rating,experience,kpis_met,awards_won,score,sum_metric,total_score]]))












