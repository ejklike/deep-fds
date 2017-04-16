import os
import h5py
import param

def get_datalist():
    assert param.trn_keys
    assert param.oot_keys
    # import data
    with h5py.File(os.path.join(param.datadir, param.fname_h5out), 'r') as h5f:
        print(' - Import training data...')
        trn_data_list = [h5f[key][:] for key in param.trn_keys]
        trn_data_list = [(data[:,:-2], data[:,-1], data[:,-2]) for data in trn_data_list]
        
        print(' - Import out-of-time data...')
        oot_data_list = [h5f[key][:] for key in param.oot_keys]
        oot_data_list = [(data[:,:-2], data[:,-1], data[:,-2]) for data in oot_data_list]

    return trn_data_list, oot_data_list


        # oot_data = (X_oot4, X_oot5, y_oot4, y_oot5, cardNo_oot4, cardNo_oot5)


