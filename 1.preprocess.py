import os
import psutil
import h5py
import pandas as pd
import numpy as np
import param

numeric_cols = [ #62 variables
    'AP_AMT', 'APP_AMT', 'APP_CNT', 
    'C1N2BC00C', 'C1N2BC00F', 'C1Z001155', 
    'DP_AMT_PRB', 'DP_AMT_SUM', 'DP_AP_CNT', 
    'DP_AP_CNT7', 'DP_BIZ_PRB', 'DP_BIZ_VEL', 
    'DP_BIZAMT_PRB', 'DP_BIZTIME_PRB', 'DP_CA_SUM', 
    'DP_CNL_CNT', 'DP_CNL_SUM', 'DP_CNL_SUM7', 
    'DP_CONT_PRB', 'DP_CONTIME_PRB', 'DP_CR_MAX', 
    'DP_CR_SUM', 'DP_DEP_CNT', 'DP_KEYIN_CNT', 
    'DP_KEYIN_PRB', 'DP_KEYIN_SUM', 'DP_KITIME_PRB', 
    'DP_MC_AGE_PRB', 'DP_MC_AMT_PRB', 'DP_MC_GTIME_PRB', 
    'DP_MC_KEYIN_PRB', 'DP_MC_SEX_PRB', 'DP_MC_WEEK_PRB', 
    'DP_RJT_CNT', 'DP_RJT_SUM', 'DP_RJT_SUM7', 
    'DP_TIME_PRB', 'DP_WDAY_PRB', 'DP_WKTIME_PRB', 
    'LOG_DP_BIZ_RT_AMT', 'LOG_DP_RJT_CNT7', 'LOG_MC_CNT', 
    'LOG_PP_VEL1', 'LOG_PP_VEL4', 'MC_AMT', 
    'MC_CNL_AMT', 'MC_CNL_CNT', 'MC_RJT_AMT', 
    'MC_RJT_CNT', 'MC_SEXAGE_PRB', 'MC_WKTIME_PRB', 
    'PP_CD_PDUR', 'PP_GAP1', 'PP_GAP3', 
    'PP_GAP4', 'PP_GAP5', 'PP_MC_RT1', 
    'PP_MC_RT2', 'PP_NTN_RT1', 'PP_NTN_RT2', 
    'SQRT_PP_VEL3', 'WOE'
]

categoricals = { #22 variables -> 185 or 180
    'CARD_SHAPE_CD':['A','B','E','H'],
    'DP_MODE_AMT':[i+1 for i in range(4)],
    'PP_LO_TM8':[i+1 for i in range(8)],
    'PP_LO_WK':[i+1 for i in range(7)],
    'PP_MC_CONT6':[i+1 for i in range(5)],
    'PP_TX_AMT':[i+1 for i in range(7)],
    'CARD_ONL_SV_YN':['Y','N'],
    'PP_MB_AGE6':[i+1 for i in range(6)],
    'CARD_SMS_SV_YN':['Y','N'],
    # 'AP_MEDIA':['K','E','S','I','O','M','Z'], #SWIPE TOTAL
    'PP_MB_SEX':[i+1 for i in range(3)],
    'CARD_HOLD_TP':['A','B'],
    'CARD_GRD_CD':['S4','S1','P9','C1','G1','B1','X1','P2','P6','T4','T3','B3','P7','B2','V1','Q8','C3','S3','Q1','C2','C4','Q9','Q6','P1','T2','Q7','P3','P4','Q4','T5','Q3','T1','P5','X','Q5','Q2','31','I1','C5','S2','P8'],
    'FALLBACK_APV_TF':['Y','N'],
    'AP_TX_TP':['A','C'],
    'AP_TX_KD':[5,6,7,8],
    'BIZGR':[i for i in range(14)],
    'FB_PPAMT':['N1','N2','N3','Y1','Y2','Y3'],
    'FB_PPAGE':['N1','N2','N3','Y1','Y2','Y3'],
    'FB_MDAMT':['N1','N2','N3','Y1','Y2','Y3'],
    'FB_MC_CNT':['N1','N2','N3','Y1','Y2','Y3'],
    'CPP_CD_YN':['Y','N']
    }

categorical_names_MS = [ #22 variables
    'PP_MB_AGE6', 'PP_LO_WK', 'PP_MC_CONT6', 
    'CPP_CD_YN', 'PP_MB_SEX', 'CARD_SHAPE_CD', 
    'PP_LO_TM8', 'BIZGR', 'FB_MDAMT', 
    'CARD_GRD_CD', 'PP_TX_AMT', 'CARD_SMS_SV_YN', 
    'FB_PPAGE', 'AP_TX_KD', 'CARD_ONL_SV_YN', 
    'FB_PPAMT', 'FB_MC_CNT', 'DP_MODE_AMT', 
    'FALLBACK_APV_TF', 'AP_MEDIA', 'CARD_HOLD_TP', 
    'AP_TX_TP']

categorical_names_IC = [ #21 variables
    'PP_MB_SEX', 'PP_TX_AMT', 'FB_PPAMT', 
    'FB_MDAMT', 'PP_LO_WK', 'FB_PPAGE', 
    'BIZGR', 'CPP_CD_YN', 'CARD_HOLD_TP', 
    'DP_MODE_AMT', 'PP_MC_CONT6', 'CARD_SMS_SV_YN', 
    'CARD_GRD_CD', 'CARD_ONL_SV_YN', 'PP_MB_AGE6', 
    'AP_TX_KD', 'FB_MC_CNT', 'CARD_SHAPE_CD', 
    'AP_TX_TP', 'FALLBACK_APV_TF', 'PP_LO_TM8']


class preprocess(object):

    def __init__(self):
        print(' - Segment:', param.segment)
        print(' - trn_keys:', param.trn_keys)
        print(' - oot_keys:', param.oot_keys)
        self.key_list = param.trn_keys + param.oot_keys
        self.memory_upperbound = 80

    def get_stat(self):
        df_list = []
        for i, yyyymm in enumerate(param.trn_keys):
            fname = os.path.join(
                param.datadir, 
                param.fname_csvin.format(yyyymm)
            )
            print('Load data from {}'.format(fname), end=': ')
            df = pd.read_csv(fname, low_memory=False)
            memory_usage = psutil.virtual_memory().percent
            print('memory usage = {}%'.format(memory_usage))
            if param.segment == 'IC':
                df = df[df.AP_MEDIA == 'I']
            elif param.segment =='MS':
                df = df[df.AP_MEDIA != 'I']
            else:
                raise Exception('you shuld set segment parameter as "IC" or "MS".')
            df_list.append(df)
            if memory_usage > self.memory_upperbound:
                print('memory usage exceeded {}%'.format(memory_usage))
                print('cannot import these data: {}'.format(param.trn_keys[(i+1):]))
                print('use only these data: {}'.format(param.trn_keys[:i+1]))
                break

        print('Calculate summary statistics...', end='')
        df = df_list[0]
        for x in df_list[1:]:
            df = df.append(x, ignore_index=True)
        stat = df[numeric_cols].describe()
        stat = stat.loc[['50%','std']].T
        stat = stat.rename(index=str, columns={'50%':'med'})
        stat.index.name = 'name'
        fname_stat = 'med_std_{}.csv'.format(param.segment)
        fname_stat = os.path.join(param.datadir, fname_stat)
        stat.to_csv(fname_stat, sep=',')
        print('done.')
        print('"{}" was created.'.format(fname_stat))
        return None
    
    def transform_data(self):
        with h5py.File(os.path.join(param.datadir, param.fname_h5out), 'a') as h5f:
            for i, yyyymm in enumerate(self.key_list):
                fname = os.path.join(
                    param.datadir, 
                    param.fname_csvin.format(yyyymm)
                )
                print('Load data {}/{}... filename: {}'.format(i+1, len(self.key_list), fname))
                save_key = yyyymm if yyyymm in param.oot_keys else None
                data = _scaling_and_dummy(fname, save_key=save_key)
                h5f.create_dataset(yyyymm, data=data)

def _scaling_and_dummy(fname, save_key=None):
    df = pd.read_csv(fname, low_memory=False, sep=param.delimiter)
    before_shape = df.shape

    if df.shape[1]==1: #delimiter did not work
        raise Exception('you shuld set correct delimiter parameter for csv files.')

    #row-wise selection
    if param.segment == 'MS':
        df = df[df.AP_MEDIA != 'I']
        med_std = pd.read_csv('./data/med_std_MS.csv')
        categoricals['AP_MEDIA'] = ['K','E','S','O','M','Z']
        categorical_names = categorical_names_MS
    elif param.segment == 'IC':
        df = df[df.AP_MEDIA == 'I']
        med_std = pd.read_csv('./data/med_std_IC.csv')
        categorical_names = categorical_names_IC
    else:
        raise Exception('you shuld set segment parameter as "IC" or "MS".')
    
    if df['CARD_NO'].dtype != np.int64 and df['CARD_NO'].dtype != np.int32:
        df['CARD_NO'] = pd.to_numeric(df['CARD_NO'].str[1:])
    card = np.array(df['CARD_NO']).reshape(-1,1)
    y = np.array(df['TARGET1']).reshape(-1,1)

    # save_key
    if save_key is not None:
        keys = df[param.key_columns]
        fname_key = os.path.join(
            param.datadir, 
            'key_{}_{}.csv'.format(param.segment, save_key))
        keys.to_csv(fname_key, sep=',')
        print(' - keys are saved to {}'.format(fname_key))

    # numeric transform (x-med)/std
    avg, std = 0, 0
    for i in range(len(med_std)):
        name, med, std = med_std.iloc[i,0], med_std.iloc[i,1], med_std.iloc[i,2]
        df[name] = (df[name]-med)/std  
        this_avg = df[name].mean()
        this_std = df[name].std()
        avg += this_avg/len(med_std)
        std += this_std/len(med_std)
        print(' - numerical variable: rescaling (x-med)/std for {}        '.format(name), flush=True, end='\r')
        # print('{}. {} : mean({}), stdev({})'.format(
        # i, name, this_avg, this_std), end='\r', flush=True)
    numeric_np = np.array(df[list(med_std.name)])
    if abs(avg) > 1 or std > 2:
        print('overall avg. of (mean, std) for all variables is ({}, {}), respectively. '.format(round(avg,2), round(std,2)))
        print('if these values are not around (0, 1), you should run get_stat() first.')
    
    # dummy transform
    for i, col in enumerate(categorical_names):
        for value in categoricals[col][:-1]:
            dummy_col = np.array(df[col]==value, dtype=int).reshape(-1, 1)
            numeric_np = np.append(numeric_np, dummy_col, axis=1)
        print(' - categorical variable: create dummy variables for {}        '.format(col), flush=True, end='\r')
    finalData = np.concatenate([numeric_np, card, y], axis=1)
    print(' - (#transaction, #fraud, %fraud) = ({}, {}, {}%), #variable = {}'.format(
        finalData.shape[0], 
        np.sum(y), 
        np.round(np.sum(y)/finalData.shape[0],4), 
        finalData.shape[1]-2))
    return finalData

if __name__ == '__main__':
    print('*** Load preprocess module...')
    preprocessor = preprocess()
    print('*** Get statistics of numerical variables from data...')
    preprocessor.get_stat()
    print('*** Preprocess data...')
    preprocessor.transform_data()