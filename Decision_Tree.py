#importing all the required modules 
#-----------------------------------------------------------------------------
#For performing different Mathematical operation we will import numpy
import numpy as np
#For loading the dataset and processing it we will use pandas
import pandas as pd
#Decision Tree calssifier is present in the sklearn module
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing

#for making a model and loading it,we will use joblib  
from sklearn.externals import joblib


#Reading the dataset
balance_data = pd.read_csv('bank1.csv', header = 0, delimiter=' *, *', engine='python')

print ("Dataset Length:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)

print ("Dataset:: ")
balance_data.head()

#Encoding the model to a specific format to fit and transform the model
le = preprocessing.LabelEncoder()
balance_data = balance_data.apply(le.fit_transform)


# "X" are the input attributes
X = balance_data.values[:, 0:16]

# "Y" is the output varibale
Y = balance_data.values[:,16]

# Splitting the into training  and testing data that is in a ration of 70:30
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.33, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

#DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
 #           max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
  #          min_samples_split=2, min_weight_fraction_leaf=0.0,
   #         presort=False, random_state=100, splitter='best')


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

#DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
     #       max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
      #      min_samples_split=2, min_weight_fraction_leaf=0.0,
       #     presort=False, random_state=100, splitter='best')


#print (clf_gini.predict([[]]))

y_pred = clf_gini.predict(X_test)
y_pred


y_pred_en = clf_entropy.predict(X_test)
y_pred_en

#Displaying the accuracy
#-----------------------------------
#Accuracy using gini
print ("Accuracy bt using gini index is :", accuracy_score(y_test,y_pred)*100)

#Accuracy using entropy
print ("Accuracy by using entropy is :", accuracy_score(y_test,y_pred_en)*100)

# save the model to disk
filename = 'finalized_model_for_entropy.sav'
filename1= 'finalized_model_for_gini.sav'
joblib.dump(clf_entropy, filename)
joblib.dump(clf_gini, filename1)

# load the model from disk i.e; Entropy
loaded_model_1 = joblib.load(filename)
result = loaded_model_1.score(X_test, y_test)

#load the model from the disk i.e; Gini
loaded_model_2 = joblib.load(filename)
result1 = loaded_model_2.score(X_test, y_test)

#Concluding the best one
if(result>result1):
    print("\nBest accuracy  is by using gini :",result*100)
else:
    print("\nBest accuracy  is by using entropy :",result1*100)
