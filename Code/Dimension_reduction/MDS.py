import scipy.io as sio
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier,ExtraTreesClassifier)
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import scale
import utils.tools as utils
from sklearn.manifold import MDS
from sklearn.metrics import roc_curve, auc
from dimensional_reduction import mds
import pandas as pd
import matplotlib.pyplot as plt

data_train=pd.read_csv(r'ST_fusion.csv')
data_=np.array(data_train)
data=data_[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(m1/2),1))
label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
shu=scale(data)
data_2=mds(shu,n_components=347)
shu=data_2
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('MDS_Y_train_347.csv')

