######################################################
# DEFINE DATA SEGMENT
#  - 'IC' or 'MS'
#  - 'yyyymm' (key) list of training and oot data
######################################################

segment = 'MS' 
# segment = 'IC'

# CONFIGURE YYYYMM KEYS
#  - TRAINING DATA AND OOT TEST DATA

trn_keys = [
    '201504', '201505','201506','201507','201508','201509',
    '201510','201511','201512','201601','201602','201603'
]
oot_keys = [
    '201604','201605'#,'201612'
]

delimiter = ',' # csv file delimiter(separater)
# delimiter = '|' # csv file delimiter(separater)

# key_columns = ['CARD_NO', 'AP_TX_DT', 'AP_TX_TP', 'AP_AW_TP', 'AP_TX_NO', 'AP_TX_KD']
key_columns = ['CARD_NO', 'AP_TX_DT', 'AP_TX_TP', 'AP_TX_KD'] # 'AP_AW_TP', 'AP_TX_NO', 

######################################################
# SELECT A MODEL TO TRAIN (OR LOAD)
# - model design: select only one, and comment the rests
# - model_id = None or previous_mode_id
#
# (caution) if you want to load specific model with `model_id`, 
# its model design should be matched to the below configuration! 
######################################################
# TRAIN = 0 #if you just want to load and check the model
TRAIN = 1 #if you want to train a new model

# model_design = 'MS-DNN1'
model_design = 'MS-DNN2'
# model_design = 'IC-DNN'
 
model_id = None #if you want to train new model
# model_id = '2017-02-14 10:50:32'

# final models
# model_id = '2017-01-18 15:01:56-a' #MS-DNN1 #final
# model_id = '2017-01-21 19:35:14-a' #MS-DNN2-1 #final
# model_id = '2017-01-21 15:54:53-a' #MS-DNN2-2 #final
# model_id = '2017-01-20 18:52:20-a' #IC-DNN #final


######################################################
# LEARNING PARAMETERS
######################################################
learning_rate_list = [0.001, 0.01, 0.01]
keeprate_list = [0.99, 0.95, 0.9, 0.75, 0.5]

batch_size_list = [1000, 2000, 3000, 5000, 10000]
batch_fraud_ratio_list = [0.01, 0.05, 0.1, 0.3, 0.5]
minor_penalty_list = [100, 20, 10, 3, 2]

development_set_size = 0.15 # between 0 and 1
num_model_per_config = 30 # train distinct N models per setting 


######################################################
# THE RESULTS OF MODEL: SELECT OPTIONS
######################################################
print_result = True
save_result = True #in 'record.csv'
save_individual_scores = False
export_weight = False


######################################################
# OTHER PARAMETERS
######################################################
USE_GPU = 1 # USE_GPU = 1 or 0

#define data directory
datadir = './data/'
#define rawdata filename(csv fileformat)
fname_csvin = 'data-swipe-{}.csv' #.format(yyyymm)
#define preprocessed data filename(h5 fileformat)
fname_h5out = 'data-{}.h5'.format(segment)

