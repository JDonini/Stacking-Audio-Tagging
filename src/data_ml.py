from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os
import pandas as pd


def remove_empty(arr):
    if(arr[-1] == ''):
        arr.pop()
    return arr


def load_data(model,fold):   
    X_train = open('../data/features/{0}/{0}-f{1}-train.csv'.format(model,fold)).read().split('\n')
    remove_empty(X_train)
    y_train = []
    X_test = open('../data/features/{0}/{0}-f{1}-test.csv'.format(model,fold)).read().split('\n')
    remove_empty(X_test)
    y_test = []
    
    files = open('../config/P-TMDB-all-id-genre.csv','r+').read().split('\n')
    Y={data.split(';')[0]:data.split(';')[1].split('-') for data in files}

    classes = set()
    for i in range(len(X_train)):
        X_train[i] = X_train[i].split(';')
        id = X_train[i][0]
        X_train[i] = X_train[i][1:]
        classes.update(Y[id])
        y_train.append(Y[id]) 

    for i in range(len(X_test)):
        X_test[i] = X_test[i].split(';')
        id = X_test[i][0]
        X_test[i] = X_test[i][1:]
        classes.update(Y[id])
        y_test.append(Y[id])

    classes = sorted(list(classes))
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])

    y_train = mlb.transform(y_train)
    y_test = mlb.transform(y_test)

    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)
    return X_train,X_test,y_train,np.array(y_test, dtype=float),classes


def load_results(feature,classifier,fold):
    predictions = open('../data/results/{0}/{1}/{0}_f{2}.pred'.format(feature,classifier,fold)).read().split('\n')
    predictions[0] = predictions[0].split(';')
    probabilities = open('../data/results/{0}/{1}/{0}_f{2}.proba'.format(feature,classifier,fold)).read().split('\n')
    probabilities[0] = probabilities[0].split(';')
    # probabilities.pop(0)
    for i in range(len(predictions)-1):
        i=i+1
        predictions[i] = np.array(predictions[i].split(';'),dtype=np.float).astype(np.int)
        probabilities[i] = np.array(probabilities[i].split(';'),dtype=np.float)

    return predictions,probabilities


def load_prediction_cnn(model,batch_size,seq_length,data_size='full'):   
    pred = open('results/{0}/cnn/b{1}_f{2}_{3}.pred'.format(model,batch_size,seq_length,data_size)).read().split('\n')[1:]
    proba = open('results/{0}/cnn/b{1}_f{2}_{3}.proba'.format(model,batch_size,seq_length,data_size)).read().split('\n')[1:]
    
    files = open('data/data_file.csv','r+').read().split('\n')
    files.pop()
    Y=[data.split(',')[1].split('-') for data in files]
    ids=[data.split(',')[2] for data in files]

    true = []
    classes = set()

    for i in range(len(pred)):
        pred[i] = pred[i].split(';')
        id = pred[i][0]
        pred[i].remove(id)
        index = ids.index(id)
        classes.update(Y[index])
        true.append(Y[index]) 
    
    for i in range(len(proba)):
        proba[i] = proba[i].split(';')
        id = proba[i][0]
        proba[i].remove(id)

    classes = sorted(list(classes))
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    true = mlb.transform(true)
    return np.array(pred, dtype=float),np.array(proba, dtype=float),true


def load_ids(model,fold):   
    X_train = open('../data/features/{0}/{0}-f{1}-train.csv'.format(model,fold)).read().split('\n')
    remove_empty(X_train)
    X_test = open('../data/features/{0}/{0}-f{1}-test.csv'.format(model,fold)).read().split('\n')
    remove_empty(X_test)
    
    train_ids = []    
    for i in range(len(X_train)):
        train_ids.append(X_train[i].split(';')[0])

    test_ids = []
    for i in range(len(X_test)):
        test_ids.append(X_test[i].split(';')[0])
       
    return train_ids,test_ids


def save_pd_result(df,model,classifier,fold,type):
    file = '{0}/{1}/{0}-f{2}.{3}'.format(model,classifier,fold,type)
    save_pd(df,'results',file)


def save_pd(df,file_type,file):
    file = '../data/{0}/{1}'.format(file_type,file)
    df.to_csv(file,sep=';',float_format='%.4f',index=False)


def rename_results_file(model,classifier,fold):
    src = '../data/results/{0}/{1}/{0}_f{2}.pred'.format(model,classifier,fold)
    des = '../data/results/{0}/{1}/{0}_f{2}.pred.bak'.format(model,classifier,fold)
    os.rename(src, dst)
    src = '../data/results/{0}/{1}/{0}_f{2}.proba'.format(model,classifier,fold)
    des = '../data/results/{0}/{1}/{0}_f{2}.proba.bak'.format(model,classifier,fold)
    os.rename(src, dst)


def load_feature_pd(feat):
    return pd.read_csv('../data/features/'+feat,sep=';',header=0,index_col=0).sort_index()


def get_classes():
    return pd.read_csv('../config/P-TMDB-all-id-genre.csv',sep=';',header=None,index_col=0).sort_index().values


def get_classes_by_id(ids):
    return pd.read_csv('../config/P-TMDB-all-id-genre.csv',sep=';',header=None,index_col=0).loc[ids].values
