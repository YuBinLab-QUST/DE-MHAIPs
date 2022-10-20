
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re
from scipy import interp
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, merge, Reshape, Lambda
from keras.layers import Flatten
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale, StandardScaler
from keras.layers import Dense, merge, Input, Dropout
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D
from tensorflow.python.keras.layers import Layer, InputSpec
from keras import backend as K
import utils.tools as utils



class MyMultiHeadAttention(Layer):
    def __init__(self,output_dim,num_head,name="attention",**kwargs):
        self.output_dim = output_dim
        self.num_head=num_head
        super(MyMultiHeadAttention,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.W=self.add_weight(name='W',shape=(self.num_head,3,input_shape[2],self.output_dim),
           initializer='uniform',
           trainable=True)
      
        self.Wo=self.add_weight(name='Wo',shape=(self.num_head*self.output_dim,self.output_dim),
        
           initializer='uniform',
           trainable=True)
        super(MyMultiHeadAttention, self).build(input_shape)  
    def call(self,x):
        for i in range(self.W.shape[0]):
            q=K.dot(x,self.W[i,0])
            k=K.dot(x,self.W[i,1])
            v=K.dot(x,self.W[i,2])
            print("q.shape", q.shape)

            print("K.permute_dimensions(k, [0, 2, 1]).shape", K.permute_dimensions(
            k, [0, 2, 1]).shape)
            
           
            e=K.batch_dot(q,K.permute_dimensions(k,[0,2,1]))
            e=e/(self.output_dim**0.5)
            e=K.softmax(e)
            print("e.shape", e.shape)
            o=K.batch_dot(e,v)
            if i ==0:
                outputs=o
            else:
                outputs=K.concatenate([outputs,o])
        z=K.dot(outputs,self.Wo)
        return z
    def compute_output_shape(self,input_shape):
          return (input_shape[0],input_shape[1],self.output_dim)

att=MyMultiHeadAttention(3,5)
print(att)



def to_class(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y

def get_shuffle(dataset,label):    
   
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label 

data_=pd.read_csv(r'Group_Lasso_FUusion_5-train.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
#label1=np.ones((int(m1/2),1))#Value can be changed
#label2=np.zeros((int(m1/2),1))
label1=np.ones((4308,1))#Value can be changed
label2=np.zeros((4308,1))
label=np.append(label1,label2)
X_=scale(data)
y_= label
X,y=get_shuffle(X_,y_)
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5


def get_RNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(LSTM(int(input_dim/2), return_sequences=True,activation = 'sigmoid'))
    model.add(Dropout(0.5))
    model.add(LSTM(int(input_dim/4), return_sequences=True,activation = 'sigmoid'))
    model.add(Dropout(0.5))
    model.add( MyMultiHeadAttention(3,5))
    model.add(Flatten())
    model.add(Dense(256, activation = 'sigmoid',name="Dense_64")) 
    model.add(Dropout(0.5)) 
    model.add(Dense(64, activation = 'sigmoid',name="Dense_16")) 
    model.add(Dropout(0.5)) 
    model.add(Dense(2, activation = 'softmax',name="Dense_2")) 
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])
    return model

[sample_num,input_dim]=np.shape(X)
out_dim=2
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
probas_rnn=[]
tprs_rnn = []
sepscore_rnn = []
skf= StratifiedKFold(n_splits=5)
for train, test in skf.split(X,y):
    time_step = 10
    features_num = input_dim
    seg_len = int(features_num / time_step)
    clf_rnn = get_RNN_model(input_dim,out_dim)
    X_train_rnn=np.reshape(X[train],(-1,1,input_dim))
    X_test_rnn=np.reshape(X[test],(-1,1,input_dim))
    clf_list = clf_rnn.fit(X_train_rnn, to_categorical(y[train]),batch_size=16, epochs=60)
    y_rnn_probas=clf_rnn.predict(X_test_rnn)
    probas_rnn.append(y_rnn_probas)
    y_class= utils.categorical_probas_to_classes(y_rnn_probas)
    
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]  
    yscore=np.vstack((yscore,y_rnn_probas))
    
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class,y[test])
    mean_fpr = np.linspace(0, 1, 150)
    fpr, tpr, thresholds = roc_curve(y[test], y_rnn_probas[:, 1])
    tprs_rnn.append(interp(mean_fpr, fpr, tpr))
    tprs_rnn[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    sepscore_rnn.append([acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc])
                            

row=ytest.shape[0]
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum_MutiAt_G_L_ST.csv')

yscore_=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore_)
yscore_sum.to_csv('yscore_sum_MutiAt_G_L_ST.csv')

scores=np.array(sepscore_rnn)
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscore_rnn.append(H1)
result=sepscore_rnn
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('MutiAt_G_L_ST .csv')

