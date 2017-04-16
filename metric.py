import numpy as np

def crossEntropy(y, yhat, c):
    eps = 1e-15
    yhat = np.clip(yhat, eps, 1-eps)
    pos = c*np.log(yhat)*y
    neg = 1*np.log(1-yhat)*(1-y)
    tmp =  (pos+neg) / (c+1)
    tmp = - np.sum(tmp) / len(yhat)
    return tmp


def eval(y_true, y_score, cardNo, cutoff=None, top_k=10000):
    #basic stat: by transaction
    data_size = len(y_true)
    card_size = len(np.unique(cardNo))
    fraud_size = int(np.sum(y_true))
    # fraud_ratio = fraud_size / data_size

    if cutoff: #if cutoff method is selected
        top_k = sum(y_score >= cutoff)

    # select top k score transactions
    top_k_indices = np.argsort(-y_score)[:top_k]
    top_k_fraud_size = np.sum(y_true[top_k_indices])

    # select all fraud cards (denominator)
    fraud_card_list = np.unique(cardNo[np.argwhere(y_true==1)]) #denominator
    top_k_card_list = np.unique(cardNo[top_k_indices])
    # select top k fraud cards (numerator)
    top_k_fraud_card_list = np.intersect1d(top_k_card_list, fraud_card_list)

    #eval by transactions
    shoot_ratio_transaction = top_k_fraud_size / fraud_size
    
    #eval by cards
    if len(fraud_card_list) > 0:
        shoot_ratio_card = len(top_k_fraud_card_list) / len(fraud_card_list)
    else:
        shoot_ratio_card = 0

    y_tmp = np.zeros(data_size)
    y_tmp[top_k_indices] = 1

    # shoot_transaction < -validation result OK 
    # precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_tmp, average='binary')
    # print(precision, recall, fscore)
    # print(shoot_ratio_transaction)

    # result = (precision, recall, fscore, 
    recall_trans = shoot_ratio_transaction*100
    recall_cards = shoot_ratio_card*100
    
    return recall_trans, recall_cards, top_k