from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np 

def process_data(path,test_size,random_state):
    data = pd.read_csv("data/creditcardfraud.zip",compression='zip')
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time','Amount'],axis=1)
    X = data.iloc[:, data.columns != 'Class'].values
    y = data.iloc[:, data.columns == 'Class'].values.flatten()
    shape=X.shape[1]
    outlier=np.unique(y,return_counts=True)[1]
    print(f"Number of features  is  {shape}")
    print("the size of the dataset is {datasize} and the total number of pozitive cases is {outliersize}".format(datasize=outlier[0],outliersize=outlier[1]))
    x_in = X[y==0]
    X_train, X_test = train_test_split(x_in, test_size=test_size, random_state=random_state)
    x_in = X_test[:492]
    x_out = X[y==1][:492]
    test_X = np.concatenate((x_in,x_out),axis=0)
    test_y = np.concatenate((np.zeros(len(x_in)),np.ones(len(x_out))))
    return X_train, test_X,test_y 
