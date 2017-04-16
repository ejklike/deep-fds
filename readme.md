# preparation

Prepare "./data" folder, which contains 
  "monthly swipe data" files with csv format

- e.g., 'data-201603.csv', 'data-201604.csv', ...


# process

1. prepare csv files of your monthly transaction data
2. modify `param.py` for your data preprocessing and training
3. run `python 1.preprocess.py` in terminal
4. run `python 2.train.py` in terminal

> (caution) IF YOU ARE NOT FAMILIAR WITH PYTHON CODE,
> JUST DO EDIT `param.py` ONLY, NOT THE REST!


# 1.preprocess.py

Input param
---
segment:    'IC' or 'MS'
month_keys: list of 'yyyymm' strings
  - trn_keys, oot_keys
delimiter:  delimiter of csv files

Output
---
a data file with h5 format
e.g., 'data-IC.h5' or 'data-MS.h5'


# 2.train.py

Input param
---
train: 1 or 0
  - 1: if you want to train a new model
  - 0: just load and check existing model (with specific model_id)
model_design: model architecture
  - MS-DNN1, MS-DNN2, IC-DNN
model_id: load previous model with model_id
  - 'None' if you want to train a new model
  
learning parameters: various settings...
  - tip1: try as many configurations as possible
  - tip2: for a setting, train 10 or more models 
          (some models show high performance, 
          and some models show low performance
           due to some randomness in training)

Output
---
several options are available to see or save scoring result, or export model
  - print_result, save_result, save_individual_scores, export_weight
  - True of False


# Other params

you can specify the below:

- data directory and filename
- use gpu or cpu