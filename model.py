import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer
import param

def get_model_config(model_design):
    model_id = param.model_id
    if model_design == 'MS-DNN1':
        model_design = multiple_dnn_jump_concat
        model_size = [[100],[50,40,30]]
        # model_id = '2017-01-18 15:01:56-a'
    elif model_design == 'MS-DNN2':
        model_design = multiple_dnn_jump_concat
        model_size = [[50],[50,30],[30,30,30]]
        # model_id = '2017-01-21 19:35:14-a'
        # model_id = '2017-01-21 15:54:53-a'
    elif model_design == 'IC-DNN':
        model_design = multiple_dnn_jump_concat
        model_size = [[200]]
        # model_id = '2017-01-20 18:52:20-a'
    else:
        raise Exception("CHOOSE CORRECT MODEL DESIGN IN 'param.py'")
    return model_design, model_size, model_id

def multiple_dnn_jump_concat(x, model_size_list, keeprate):
    """
    model_size_list = [[50,40,30], [40,30], [20]]
    """
    model_list = []
    model_list.append(x)
    for model_no, model_size in enumerate(model_size_list):
        h = x
        for i, hidden_size in enumerate(model_size):
            # print(h.get_shape())
            h = fully_connected(h, hidden_size, 
                scope='dnn%s/h%s'%(model_no+1, i+1),
                weights_initializer=xavier_initializer(),
                biases_initializer=xavier_initializer(),
                activation_fn = tf.nn.sigmoid,
                trainable=True)
            h = tf.nn.dropout(h, keeprate)
            model_list.append(h)
    h = tf.concat(1,model_list)
    out = fully_connected(h, 2, scope='out',
        weights_initializer=xavier_initializer(),
        biases_initializer=xavier_initializer(),
        activation_fn = tf.nn.softmax,
        trainable=True)
    return out

def multiple_dnn_jump_concat_no_X(x, model_size_list, keeprate):
    """
    model_size_list = [[50,40,30], [40,30], [20]]
    """
    model_list = []
    # model_list.append(x)
    for model_no, model_size in enumerate(model_size_list):
        h = x
        for i, hidden_size in enumerate(model_size):
            # print(h.get_shape())
            h = fully_connected(h, hidden_size, 
                scope='dnn%s/h%s'%(model_no+1, i+1),
                weights_initializer=xavier_initializer(),
                biases_initializer=xavier_initializer(),
                activation_fn = tf.nn.sigmoid,
                trainable=True)
            h = tf.nn.dropout(h, keeprate)
            model_list.append(h)
    h = tf.concat(1,model_list)
    out = fully_connected(h, 2, scope='out',
        weights_initializer=xavier_initializer(),
        biases_initializer=xavier_initializer(),
        activation_fn = tf.nn.softmax,
        trainable=True)
    return out

def multiple_dnn_jump_concat_no_jump_no_X(x, model_size_list, keeprate):
    """
    model_size_list = [[50,40,30], [40,30], [20]]
    """
    model_list = []
    # model_list.append(x)
    for model_no, model_size in enumerate(model_size_list):
        h = x
        for i, hidden_size in enumerate(model_size):
            # print(h.get_shape())
            h = fully_connected(h, hidden_size, 
                scope='dnn%s/h%s'%(model_no+1, i+1),
                weights_initializer=xavier_initializer(),
                biases_initializer=xavier_initializer(),
                activation_fn = tf.nn.sigmoid,
                trainable=True)
            h = tf.nn.dropout(h, keeprate)
        model_list.append(h) #un-indent
    h = tf.concat(1,model_list)
    out = fully_connected(h, 2, scope='out',
        weights_initializer=xavier_initializer(),
        biases_initializer=xavier_initializer(),
        activation_fn = tf.nn.softmax,
        trainable=True)
    return out


def multiple_dnn_jump_concat_no_jump(x, model_size_list, keeprate):
    """
    model_size_list = [[50,40,30], [40,30], [20]]
    """
    model_list = []
    model_list.append(x)
    for model_no, model_size in enumerate(model_size_list):
        h = x
        for i, hidden_size in enumerate(model_size):
            # print(h.get_shape())
            h = fully_connected(h, hidden_size, 
                scope='dnn%s/h%s'%(model_no+1, i+1),
                weights_initializer=xavier_initializer(),
                biases_initializer=xavier_initializer(),
                activation_fn = tf.nn.sigmoid,
                trainable=True)
            h = tf.nn.dropout(h, keeprate)
        model_list.append(h) #un-indent
    h = tf.concat(1,model_list)
    out = fully_connected(h, 2, scope='out',
        weights_initializer=xavier_initializer(),
        biases_initializer=xavier_initializer(),
        activation_fn = tf.nn.softmax,
        trainable=True)
    return out