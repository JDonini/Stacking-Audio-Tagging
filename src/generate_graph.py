import matplotlib.pyplot as plt


def generate_acc_graph(history, model_name, model_stage):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(model_name + model_stage)
    plt.close()


def generate_loss_graph(history, model_name, model_stage):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(model_name + model_stage)
    plt.close()


def generate_auc_roc_graph(history, model_name, model_stage):
    plt.plot(history.history['auc_roc'])
    plt.plot(history.history['val_auc_roc'])
    plt.title('Model AUC - ROC')
    plt.ylabel('AUC - ROC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(model_name + model_stage)
    plt.close()


def generate_auc_pr_graph(history, model_name, model_stage):
    plt.plot(history.history['auc_pr'])
    plt.plot(history.history['val_auc_pr'])
    plt.title('Model AUC - PR')
    plt.ylabel('AUC - PR')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(model_name + model_stage)
    plt.close()


def generate_hamming_loss_graph(history, model_name, model_stage):
    plt.plot(history.history['hamming_loss'])
    plt.plot(history.history['val_hamming_loss'])
    plt.title('Model Hamming Loss')
    plt.ylabel('Hamming Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(model_name + model_stage)
    plt.close()


def generate_ranking_loss_graph(history, model_name, model_stage):
    plt.plot(history.history['ranking_loss'])
    plt.plot(history.history['val_ranking_loss'])
    plt.title('Ranking Loss')
    plt.ylabel('Ranking Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(model_name + model_stage)
    plt.close()
