This repo contains 3 ml models to implement binary classification for classifying messages as spam or not spam
The 3 models compared are:
1 - Naive Bayes
2 - SVM
3 - XGBoost

The dataset employed is justmarkham's sms dataset: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

After ensuring all dependencies are installed via pip and loading the dataset, the 3 models can be run. I got the following outputs for the models running natively in jupyter notebook:

Naive Bayes:
-------------------------------------------------------------------------------------------------------------------
	Accuracy: 0.97847533632287

	Classification Report:
               precision    recall  f1-score   support

           0       0.98      1.00      0.99       966
           1       1.00      0.84      0.91       149

    accuracy                           0.98      1115
   macro avg       0.99      0.92      0.95      1115
weighted avg       0.98      0.98      0.98      1115


	Confusion Matrix:
 	[[966   0]
 	[ 24 125]]
--------------------------------------------------------------------------------------------------------------------

SVM:
--------------------------------------------------------------------------------------------------------------------
	Accuracy: 0.9901345291479821
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       966
           1       0.99      0.93      0.96       149

    accuracy                           0.99      1115
   macro avg       0.99      0.97      0.98      1115
weighted avg       0.99      0.99      0.99      1115
---------------------------------------------------------------------------------------------------------------------

XGBoost:
---------------------------------------------------------------------------------------------------------------------
	Accuracy: 0.9757847533632287
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       966
           1       0.96      0.85      0.90       149

    accuracy                           0.98      1115
   macro avg       0.97      0.92      0.95      1115
weighted avg       0.98      0.98      0.98      1115
---------------------------------------------------------------------------------------------------------------------
This was just a super basic implementation. If you want to suggest any improvements, please feel free to do so!