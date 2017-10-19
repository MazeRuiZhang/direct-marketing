import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", size = "xx-large")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_ROC(fpr, tpr, title='ROC curve'):
    lw = 2
    plt.plot(fpr, tpr, color='red',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

def plot_precision_recall():
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

mm_data = pd.read_csv(".\data\directMarketing.csv", delimiter=",")
# Check the exist of NA values
mm_data.info()
# Make sure the "saleSizeCode" attribute is category
mm_data["saleSizeCode"].value_counts()
# Get the total time span and drop no longer needed columns
mm_data["firstYear"] = np.floor(mm_data["firstDate"] / 100)
mm_data["firstMonth"] = mm_data["firstDate"] - mm_data["firstYear"] * 100
mm_data["lastYear"] = np.floor(mm_data["lastDate"] / 100)
mm_data["lastMonth"] = mm_data["lastDate"] - mm_data["firstYear"] * 100
mm_data["timeSpan"] = (mm_data["lastYear"] - mm_data["firstYear"]) * 12 + (mm_data["lastMonth"] - mm_data["firstMonth"])
mm_data = mm_data.drop(["firstDate", "lastDate", "firstYear","firstMonth","lastYear","lastMonth"], 1)
# Encode the star customer's indicator as 1
mm_data['starCustomer'] = mm_data['starCustomer'].map({'X': 1, '0': 0})
# Encode the sale size as numeric from 1:4
mm_data['saleSizeCode'] = mm_data['saleSizeCode'].map({'D': 1, 'E': 2,'F': 3, 'G': 4})
corr_matrix = mm_data.corr()
print(corr_matrix)
# As the lastSale and avrSale have a strong correlation as 0.8, I decide to drop one of them
mm_data = mm_data.drop(["lastSale"], 1)
train_set, test_set = train_test_split(mm_data, test_size=0.3, random_state=42)
mm_data = train_set.drop("class", axis=1)
mm_labels = train_set["class"].copy()
n_classes = [0,1]
# Examine the data set with histograms to discover any suspicious pattern
# The avrSale and lastSale are very fat tailed, and amount is fat tailed
mm_data.hist(bins=50, figsize=(20,15))
plt.show()

# Standardize the data set by StandardScaler method
mm_data_scaler = StandardScaler().fit_transform(mm_data)
y_test = test_set["class"]
x_test = test_set.drop(["class"], 1)

# Building the Decision Tree Model
clf_tree = DecisionTreeClassifier(random_state=42, max_depth = 3, max_leaf_nodes = 10)
clf_tree.fit(mm_data, mm_labels)
y_pred = clf_tree.predict(x_test)
print("The accuracy score of the decision tree model is: %5.3f." % accuracy_score(y_test, y_pred))
y_score = clf_tree.predict_proba(x_test)[:,1]
print("The area under the ROC curve for decision tree model is: %5.3f." % roc_auc_score(y_test, y_score))

# Plot the confusion matrix
np.set_printoptions(precision=2)
cnf_matrix_tree = confusion_matrix(y_test, y_pred)
plt.figure(1, figsize=(16,12),dpi=320)
plt.subplot(221)
plot_confusion_matrix(cnf_matrix_tree, classes=n_classes, normalize=True,
                      title='Normalized confusion matrix of decision tree model')

# Compute micro-average ROC curve and ROC area
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_pred, pos_label=1)
roc_auc = auc(fpr_tree, tpr_tree)

# Plot the ROC curve
plt.figure(2, figsize=(16,12), dpi=320)
plt.subplot(221)
plot_ROC(fpr_tree, tpr_tree, title="ROC of the Decision Tree Model")

# Plot the recision_recall curve
plt.figure(3, figsize=(16,12), dpi=160)
plt.subplot(221)
average_precision = average_precision_score(y_test, y_score)
plot_precision_recall()
plt.title('2-class Precision-Recall curve for decision tree model: AUC={0:0.2f}'.format(average_precision), size = 10)

# SVM Model
clf_svm = svm.SVC(C=1, kernel="linear", probability=True, random_state=42)
clf_svm.fit(mm_data_scaler, mm_labels)
y_pred = clf_svm.predict(StandardScaler().fit_transform(x_test))
print("The accuracy score of the SVM model is: %5.3f." % accuracy_score(y_test, y_pred))
y_score = clf_svm.predict_proba(x_test)[:,1]
print("The area under the ROC curve for SVM model is: %5.3f." % roc_auc_score(y_test, y_score))

# Plot the confusion matrix
cnf_matrix_svm = confusion_matrix(y_test, y_pred)
plt.figure(1)
plt.subplot(222)
plot_confusion_matrix(cnf_matrix_svm, classes=n_classes, normalize=True, title='Normalized confusion matrix of SVM model')

# Plot the ROC curve
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred, pos_label=1)
roc_auc = auc(fpr_svm, tpr_svm)

plt.figure(2)
plt.subplot(222)
plot_ROC(fpr_svm, tpr_svm, title="ROC of the SVM Model")

# Plot the recision_recall curve
plt.figure(3)
plt.subplot(222)
average_precision = average_precision_score(y_test, y_score)
plot_precision_recall()
plt.title('2-class Precision-Recall curve for SVM model: AUC={0:0.2f}'.format(average_precision), size = 10)

# Naive Bayes Model
clf_gnb = GaussianNB()
clf_gnb.fit(mm_data, mm_labels)
y_pred = clf_gnb.predict(x_test)
print("The accuracy score of the naive Bayes model is: %5.3f." % accuracy_score(y_test, y_pred))
y_score = clf_gnb.predict_proba(x_test)[:,1]
print("The area under the ROC curve for naive Bayes model is: %5.3f." % roc_auc_score(y_test, y_score))

# Plot the confusion matrix
cnf_matrix_gnb = confusion_matrix(y_test, y_pred)
plt.figure(1)
plt.subplot(223)
plot_confusion_matrix(cnf_matrix_gnb, classes=n_classes, normalize=True,
                      title='Normalized confusion matrix of naive Bayes model')

# Plot the ROC curve
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_pred, pos_label=1)
roc_auc = auc(fpr_gnb, tpr_gnb)
plt.figure(2)
plt.subplot(223)
plot_ROC(fpr_gnb, tpr_gnb, title="ROC of the Naive Bayes Model")

# Plot the recision_recall curve
plt.figure(3)
plt.subplot(223)
average_precision = average_precision_score(y_test, y_score)
plot_precision_recall()
plt.title('2-class Precision-Recall curve for naive Bayes model: AUC={0:0.2f}'.format(average_precision), size = 10)

# Logistic Regression Model
clf_logit = LogisticRegression(penalty="l2", C=1, random_state=42)
clf_logit.fit(mm_data, mm_labels)
y_pred = clf_logit.predict(x_test)
print("The accuracy score of the logistic regression model is: %5.3f." % accuracy_score(y_test, y_pred))
y_score = clf_logit.predict_proba(x_test)[:,1]
print("The area under the ROC curve for logistic regression model is: %5.3f." % roc_auc_score(y_test, y_score))

# Plot the confusion matrix
cnf_matrix_logit = confusion_matrix(y_test, y_pred)
plt.figure(1)
plt.subplot(224)
plot_confusion_matrix(cnf_matrix_logit, classes=n_classes, normalize=True,
                      title='Normalized confusion matrix of naive Bayes model')

# Plot the ROC curve
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred, pos_label=1)
roc_auc = auc(fpr_logit, tpr_logit)

plt.figure(2)
plt.subplot(224)
plot_ROC(fpr_logit, tpr_logit, title="ROC of the Logistic Regression Model")

# Plot the recision_recall curve
plt.figure(3)
plt.subplot(224)
average_precision = average_precision_score(y_test, y_score)
plot_precision_recall()
plt.title('2-class Precision-Recall curve for logistic regression model: AUC={0:0.2f}'.format(average_precision), size = 10)

plt.figure(1).savefig('Confusion Matrix.png')
plt.figure(2).savefig('ROC.png')
plt.figure(3).savefig('Precision-Recall.png')
plt.close('all')
