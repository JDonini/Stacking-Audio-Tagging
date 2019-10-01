import numpy as np
import pandas as pd
from data_ml import load_data,load_results,load_prediction_cnn
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, roc_auc_score, average_precision_score, multilabel_confusion_matrix


def accuracy_ml(y_pred,y_true):
    y = y_true == 1

    y_t = [[index for index, value in enumerate(data) if value == 1] for data in y_true]
    y_p = [[index for index, value in enumerate(data) if value == 1] for data in y_pred]

    acc = 0
    for i in range(len(y_t)):
        s,t = set(y_t[i]),set(y_p[i])
        union = s.union(t)
        intersection = s.intersection(t)
        acc += (len(intersection)/len(union))
    return acc/len(y_t)


def precision_ml(y_pred,y_true):
    y_t = [[index for index, value in enumerate(data) if value == 1] for data in y_true]
    y_p = [[index for index, value in enumerate(data) if value == 1] for data in y_pred]
    
    prec = 0
    for i in range(len(y_t)):
        s,t = set(y_t[i]),set(y_p[i])
        intersection = s.intersection(t)
        if(len(t) != 0):
            prec += (len(intersection)/len(t))
    return prec/len(y_t)    


def recall_ml(y_pred,y_true):
    y_t = [[index for index, value in enumerate(data) if value == 1] for data in y_true]
    y_p = [[index for index, value in enumerate(data) if value == 1] for data in y_pred]
 
    recall = 0

    for i in range(len(y_t)):
        s,t = set(y_t[i]),set(y_p[i])
        intersection = s.intersection(t)
        if(len(s) != 0):
            recall += (len(intersection)/len(s))
    return recall/len(y_t)


def metrics_cnn(model, batch_size,seq_length,data_size='full'):
    y_pred,y_proba,y_test = load_prediction_cnn(model,batch_size,seq_length,data_size)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    hl_score = hamming_loss(y_pred,y_test)
    acc_score = accuracy_score(y_pred,y_test)
    acc_ml = accuracy_ml(y_pred,y_test)
    prec_ml = precision_ml(y_pred,y_test)
    rec_ml = recall_ml(y_pred,y_test)
    fmeasure_score = f1_score(y_pred,y_test, average='micro')
    roc_score = roc_auc_score(y_test,y_proba, average='micro')
    pr_score = average_precision_score(y_test,y_proba,average='micro')

    print('{0}-{1};{2:.4f};{3:.4f};{4:.4f};{8:.4f};{9:.4f};{5:.4f};{6:.4f};{7:.4f}'.format(model,
                                                                                           'cnn', hl_score, acc_score, acc_ml, fmeasure_score, roc_score, pr_score, prec_ml, rec_ml))


def metrics_database_patch(y_true,patch):
    database = np.array(y_true, dtype=np.float)
    num_amostras = len(database)
    df = pd.DataFrame(database).drop_duplicates(keep='first')
    div_rot = df.shape[0]
    prop_div_rot = df.shape[0]/num_amostras    
    database = database.sum(axis=0)
    card_rot = database.sum()/num_amostras
    den_rot = card_rot/18
    classes = ';'.join(database.astype(str))

    print('{6};{0};{1};{2};{3};{4};{5}'.format(num_amostras,div_rot,prop_div_rot,card_rot,den_rot,classes,patch))


def metrics_database(batch_size,seq_length,data_size='full'):
    _,_,y_train,y_test,classes = load_data('c3d',batch_size,seq_length,data_size=data_size)
    
    print(';Número de Amostras;Diversidade de rótulos;Proporção da diversidade de rótulos;Cardinalidade de rótulos;Densidade de rótulos;'+';'.join(classes))

    metrics_database_patch(y_train,'Treino')
    metrics_database_patch(y_test,'Teste')
    metrics_database_patch(np.concatenate((y_train,y_test),axis=0),'Total')


def gen_metric_file():
    arr = ['Feature; Classifier; Type; Hamming Loss; Subset Accuracy; Accuracy;Precision; Recall; F1 Score; AUC ROC; AUC PR']
    open('../data/results/P-TMDB.csv','w+').write('\n'.join(arr))


if __name__ == '__main__':
    gen_metric_file()
    # models = ['ctt','c3d','lrcn']
    # classifiers = ['binaryrelevance_svm','binaryrelevance_mlp','binaryrelevance_dt','classifierchain_svm','classifierchain_mlp','classifierchain_dt','mlknn']
    # classifiers_ml = ['binaryrelevance','classifierchain']
    # data_sizes = ['4k','35k']
    # # ctt,c3d,lrcn
    # # binaryrelevance_svm,binaryrelevance_mlp,binaryrelevance_dt,classifierchain_svm,classifierchain_mlp,classifierchain_dt,mlknn

    # batch_size = 2
    # seq_length = 120

    # data_size = 'p-tmdb'

    # classifier = 'binaryrelevance_svm'

    # print(';Hamming Loss;Subset Accuracy;Accuracy;Precision;Recall;F1 Score;AUC ROC;AUC PR')

    # for model in models:
    #     # for classifier in classifiers:
    #         # metrics_cnn(model=model, classifier=classifier, batch_size=batch_size,seq_length=seq_length,data_size=data_size)
    #     metrics_cnn(model=model, batch_size=batch_size,seq_length=seq_length,data_size=data_size)
    # # for model in models:
    #     metrics_cnn(model=model, batch_size=batch_size,seq_length=seq_length,data_size=data_size)
    # metrics_database(batch_size=batch_size,seq_length=seq_length,data_size=data_size)

    # metrics_cnn(model='c3d', batch_size=batch_size,seq_length=seq_length,data_size='4k')
    # model = 'lrcn'
    # prediction(model='lrcn', classifier='classifierchain_svm', batch_size=batch_size,seq_length=seq_length,data_size='4k')
    # prediction(model=model, classifier='classifierchain_svm', batch_size=batch_size,seq_length=seq_length,data_size='35k')
