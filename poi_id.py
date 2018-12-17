#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier

# Packages and functions added
import functions_poi as fp
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### features list took in all the features and after analysis they would be selected further.

features_list = ['poi' ,# poi label #
                 'salary',# financial features #
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive',
                 'restricted_stock',
                 'director_fees',
                 'to_messages',# email features #
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    

### Task 2: Remove outliers

data_dict.pop( "TOTAL", 0 )
data_dict.pop("LOCKHART EUGENE E",0)
### Task 3: Create new feature(s)
data_dict=fp.salary_Bonus(data_dict)
data_dict=fp.ratio_fm(data_dict)
data_dict=fp.ratio_tm(data_dict)

features_list+=['bonus_salary_sum','poi_proportion_fm','poi_proportion_tm']
## Uncomment to check if features added are correct
## print data_dict["SKILLING JEFFREY K"]

### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

## SelectKbest
# Change k=6 if you want to know why I chose 5

K_best=SelectKBest(f_classif,k=5)
features_kbest=K_best.fit_transform(features,labels)
# print "Shape of features after applying SelectKBest->",features_kbest.shape
# SelectKBEst scores,rounded to 2 decimal places
feature_scores=['%.2f'%elem for elem in K_best.scores_]

features_selected_tuple=[(features_list[i+1],feature_scores[i])for i in K_best.get_support(indices=True)]

features_selected_tuple=sorted(features_selected_tuple,key=lambda feature:float(feature[1]),reverse=True)
features_list_selected=[(features_list[i+1])for i in K_best.get_support(indices=True)]

print features_selected_tuple

# Updating the features and features_list
features=features_kbest
features_list=['poi']+features_list_selected

# check if features are of your desired choice
print features_list


# Provided to give you a starting point. Try a variety of classifiers.
'''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf,my_dataset,features_list)

from sklearn.ensemble import AdaBoostClassifier
clf1 = AdaBoostClassifier(n_estimators=50)
test_classifier(clf1,my_dataset,features_list)

from sklearn.neighbors import KNeighborsClassifier
clf3 = KNeighborsClassifier(n_neighbors=4)
test_classifier(clf3,my_dataset,features_list)

from sklearn.ensemble import ExtraTreesClassifier
clf5= ExtraTreesClassifier()
test_classifier(clf5,my_dataset,features_list)
'''

clf2= DecisionTreeClassifier()
test_classifier(clf2,my_dataset,features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)




estimator = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=1, random_state=0)
estimator.fit(features_train,labels_train)
y_pred = estimator.predict(features_test)




from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


parameters={'max_features':['auto','log2'],'max_depth':[2,5,10,15],'max_leaf_nodes':[10,100]}
model=DecisionTreeClassifier()
gs_cv=GridSearchCV(model,parameters,scoring='f1',cv=2)
gs_cv.fit(features_train,labels_train)
# print gs_cv.grid_scores_

clf=gs_cv.best_estimator_
pred=clf.predict(features_test)
test_classifier(clf,my_dataset,features_list,folds=1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
