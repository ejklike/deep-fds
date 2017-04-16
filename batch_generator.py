import numpy as np

def binary_to_dummies(arr):
    arr = np.reshape(arr, (arr.shape[0],1))
    return np.concatenate((1-arr, arr), axis=1)

def train_test_split(trn_data_list, test_size=0.1, overlap_fraud_ratio=0):
    # trn_data : X,y,c in trn_data_list
    norm_data_idx_list = [ np.argwhere(y==0)  for X,y,c in trn_data_list]
    fraud_data_list = [ np.concatenate([X[y==1],y[y==1].reshape(-1,1),c[y==1].reshape(-1,1)],1) 
        for X,y,c in trn_data_list]
    fraud_data = np.concatenate(fraud_data_list, axis=0)
 
    norm_size = sum([len(X) for X in norm_data_idx_list])
    fraud_size = len(fraud_data)
    data_size = norm_size + fraud_size

    test_norm_size = int(norm_size * test_size)
    test_fraud_size = int(fraud_size * test_size)

    ##### overlap
    overlap_fraud_size = int(fraud_size * overlap_fraud_ratio)

    print(' - split fraud data')
    #split fraud data
    fraud_data = np.random.permutation(fraud_data)
    # test_fraud_data = fraud_data[:test_fraud_size]
    # train_fraud_data = fraud_data[test_fraud_size:]
    test_fraud_data = fraud_data[:test_fraud_size+overlap_fraud_size] ##### overlap
    train_fraud_data = fraud_data[test_fraud_size:]
    train_fraud_data = train_fraud_data[:,:-2] ######fraud trn data done

    print(' - split norm data')
    #split norm data
    norm_mapper = []
    for month, norm_idx_list in enumerate(norm_data_idx_list):
        for subidx in norm_idx_list:
            norm_mapper.append([month, subidx])
    norm_mapper = np.random.permutation(norm_mapper)
    norm_mapper_test = norm_mapper[:test_norm_size]
    norm_mapper_train = norm_mapper[test_norm_size:]
    # mapper_train = mapper_train[mapper_train[:,0].argsort()]
    
    # print(list(len(x) for x in mapper_train_list))

    print(' - final test data')
    #final test data
    test_data = test_fraud_data
    norm_mapper_test_list = [ norm_mapper_test[norm_mapper_test[:,0]==i,:][:,1] for i in range(len(norm_data_idx_list))]
    for i, mapper_i in enumerate(norm_mapper_test_list):
        # X, y, c = 
        Xyc = np.concatenate(\
                (trn_data_list[i][0][mapper_i], 
                trn_data_list[i][1][mapper_i].reshape(-1,1), 
                trn_data_list[i][2][mapper_i].reshape(-1,1)), 1)
        test_data = np.append(test_data, Xyc, axis=0)
    X_tst, y_tst, cardNo_tst = test_data[:,:-2], test_data[:,-2], test_data[:,-1]
    # print(train_fraud_data.shape)
    
    return (norm_mapper_train, train_fraud_data), (X_tst, y_tst, cardNo_tst)

def trn_batch_iterator(trn_data_list, norm_mapper_train, train_fraud_data, num_batch=1000, batch_size=300, fraud_ratio=0.3):

    # train (trn_batch)
    norm_mapper, fraud_X = norm_mapper_train, train_fraud_data
    norm_size, fraud_size = len(norm_mapper), len(fraud_X)

    batch_fraud_size = min(int(fraud_ratio*batch_size), fraud_size)
    batch_norm_size = batch_size - batch_fraud_size 

    # print(batch_fraud_size, batch_norm_size, fraud_ratio, batch_size, num_batch, fraud_size, norm_size) 

    for i in range(num_batch):
        fraud_indices = np.random.choice(fraud_size, batch_fraud_size)
        fraud_batch_X = fraud_X[fraud_indices]

        norm_indices = np.random.choice(norm_size, batch_norm_size)
        norm_batch_mapper = norm_mapper[norm_indices]
        norm_batch_mapper_list = [ norm_batch_mapper[norm_batch_mapper[:,0]==i,:][:,1] 
                for i in np.unique(norm_mapper[:,0])]

        fraud_batch_Y = binary_to_dummies(np.ones(len(fraud_indices)))
        norm_batch_Y = binary_to_dummies(np.zeros(len(norm_indices)))

        # print(len(norm_mapper[:,0]))
        # print(np.unique(norm_mapper[:,0]))
        
        batch_X = fraud_batch_X
        for i, mapper_i in enumerate(norm_batch_mapper_list):
            batch_X = np.append(batch_X, trn_data_list[i][0][mapper_i], axis=0)

        batch_Y = np.concatenate((fraud_batch_Y, norm_batch_Y), axis=0)
        batch_X = batch_X.reshape([-1, batch_X.shape[1]])
        
        yield (batch_X, batch_Y)


# scoring batch generator
def tst_batch(data_X, batch_size=10000):
    data_size = len(data_X)
    n_visible = data_X.shape[1]
    num_batch = int(data_size/batch_size) + 1

    for i in range(num_batch):
        start_index = i * batch_size
        end_index = min((i+1) * batch_size, data_size)
        
        batch_X = data_X[start_index:end_index]
        batch_X = batch_X.reshape([-1, n_visible])
        
        yield batch_X

