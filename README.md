# Direct-marketing
Data mining techniques to build customer profile models on predicting their subscription in the most recent campaign
## Requirements
The python program should be interpreted in the python 3.X environment
The modules required are: sklearn, numpy, pandas, matplotlib, itertools
## Installation
Place the MultiMagazine.py at the working directory and data set of directMarketing.csv at the working directory’s subfolder as “.\data\”

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
Load Data
pd.read_csv(".\data\directMarketing.csv", delimiter=",")
## Usage
This is a python program that takes no arguments. To modify functions, one can edit the program at corresponding part.
By running the main program, the output will be returned in the console and figures saved as “MultiMagazine”, “MultiMagazine” and “MultiMagazine” at the working directory.

