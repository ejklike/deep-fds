import os
from time import time
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.layers import optimize_loss
import numpy as np
import param
import batch_generator
import model
import metric
import dataio

class Model(object):
    def __init__(self,  
            model_design=model.multiple_dnn_jump_concat, 
            model_size=[[50], [50,30], [30,30,30]], 
            model_id=None):
        
        self.model_design = model_design
        self.model_name = str(model_design).split(' ')[1]
        self.model_size = model_size
        if model_id:
            self.model_id = model_id
            assert param.TRAIN == 0
        else:
            self.model_id = datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
            self.checkinterval = 50
            self.stopinterval = 1000

        print('*** MODEL CONFIG: model parameter')
        template = (
            ' - model_size = {}\n'
            ' - model_name = {}\n'
            ' - model_id = {}'
        )
        print(template.format(self.model_size, self.model_name, self.model_id))

        self.batch_size = None
        self.learning_rate = None
        self.keeprate = None
        self.batch_fraud_ratio = None
        self.minor_penalty = None

        #reset graph
        tf.reset_default_graph()
        #placeholder
        if param.segment == 'IC':
            n_visible = 180
        elif param.segment == 'MS':
            n_visible = 185
        self._X = tf.placeholder(tf.float32, shape=(None, n_visible))
        self._Y = tf.placeholder(tf.float32, shape=(None, 2))
        self._keeprate = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        # build model
        self.predictor = self.model_design(self._X, self.model_size, self._keeprate)

        #define session
        cpu_conf = tf.ConfigProto(
            device_count = {'GPU': param.USE_GPU}
        )
        self.sess = tf.Session(config=cpu_conf)

        #define tf saver
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.save_dir = './checkpoints/%s/' %(self.model_id)
        self.save_path = os.path.join(self.save_dir, 'best_validation')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            tf.set_random_seed(np.random.randint(100))
            tf.initialize_all_variables().run(session=self.sess)
            self.saver.save(sess=self.sess, save_path=self.save_path)

    def parameter_setting(self, batch_size=1000, 
        learning_rate=0.01, keeprate=0.95, 
        batch_fraud_ratio=0.1, minor_penalty=10):

        self.test_size = param.development_set_size
        self.overlap_fraud_ratio = 0
        
        self.num_batch = 10000000
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keeprate = keeprate
        self.batch_fraud_ratio = batch_fraud_ratio
        self.minor_penalty = minor_penalty

        print('*** MODEL CONFIG: learning parameter')
        template = (
            ' - batch_size = {}\n'
            ' - learning_rate = {}\n'
            ' - keeprate = {}\n'
            ' - batch_fraud_ratio = {}\n'
            ' - minor_penalty = {}'
        ).format(self.batch_size, self.learning_rate, self.keeprate, self.batch_fraud_ratio, self.minor_penalty)
        print(template)
        return model

    def train(self, trn_data_list,
        oot_monitoring=False, oot_data_list=None):
        
        assert len(trn_data_list) > 0

        #data preprocess
        print('*** data split....')
        (norm_mapper_train, train_fraud_data), (X_val, y_val, cardNo_val) = \
            batch_generator.train_test_split(trn_data_list, test_size=self.test_size, overlap_fraud_ratio=self.overlap_fraud_ratio)
        print('*** batch generation....')
        batch_iterator = batch_generator.trn_batch_iterator( \
            trn_data_list, norm_mapper_train, train_fraud_data,
            num_batch=self.num_batch, batch_size=self.batch_size, fraud_ratio=self.batch_fraud_ratio)
        
        def weighted_softmax_cross_entropy_with_logits(learner, _Y, minor_penalty):
            class_weight = tf.constant(
                [1/(minor_penalty+1), minor_penalty/(minor_penalty+1)])
            _Y = tf.mul(class_weight, _Y)
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(learner, _Y))
        cost = weighted_softmax_cross_entropy_with_logits(self.predictor, self._Y, self.minor_penalty)

        # def _learning_rate_decay_fn(learning_rate, gloabl_step):
        #     return tf.train.exponential_decay(
        #         learning_rate, 
        #         global_step, 
        #         decay_steps=50, 
        #         decay_rate=0.95)
        optimizer = optimize_loss(
            loss=cost,
            global_step=self.global_step,
            # [Adam, SGD, Momentum, RMSProp, Ftrl, Adagrad]
            # optimizer='Ftrl',
            # optimizer='RMSProp',
            optimizer='Adam',
            # learning_rate_decay_fn = _learning_rate_decay_fn,
            clip_gradients=5.0,
            learning_rate=self.learning_rate)

        #start session: training and scoring
        
        #initialize all variable
        tf.set_random_seed(np.random.randint(100))
        tf.initialize_all_variables().run(session=self.sess)
        
        ########### TRAIN THIS MODEL
        ########### until early stopping condition is satisfied
        print('*** training start...')
        best_val_cost = 1e5 #for this model
        best_val_num = -1
        try:
            cost_sum = 0
            for num, (batch_X, batch_Y) in enumerate(batch_iterator, 1):
                _, c = self.sess.run([optimizer, cost], feed_dict={
                        self._X:batch_X, self._Y:batch_Y, self._keeprate:self.keeprate})
                cost_sum += c
                # checkpoint
                if num % self.checkinterval == 0:
                    avg_cost = cost_sum / num
                    # validation in training
                    def _predict(X):
                        Y_pred = np.empty((0,2),int)
                        for batch_X in batch_generator.tst_batch(X):
                            batch_Y_pred = self.sess.run(self.predictor, feed_dict={self._X:batch_X, self._keeprate:1})
                            Y_pred = np.append(Y_pred, batch_Y_pred, axis=0)
                        y_pred = Y_pred[:,1]
                        return y_pred
                    yhat_val = _predict(X_val)
                    this_val_cost = metric.crossEntropy(y_val, yhat_val, self.minor_penalty)
                    # earlystopping
                    early_stop_condition = ( num - best_val_num > self.stopinterval )
                    if early_stop_condition: 
                        self.saver.restore(sess=self.sess, save_path=self.save_path)
                        break
                    # this step is the best
                    if this_val_cost < best_val_cost: 
                        best_val_num = num
                        best_val_cost = this_val_cost
                        self.saver.save(sess=self.sess, save_path=self.save_path)
                        template = (
                            " - temporal best model at iter {}, "
                            "tst_cost : {:.4f}, this_trn_cost: {:.4f}, avg_trn_cost: {:.4f}"
                        )
                        print(template.format(num, this_val_cost, c, avg_cost), flush=True)
                    else:
                        template = (
                            " - [status] iter {}, "
                            "tst_cost : {:.4f}, this_trn_cost: {:.4f}, avg_trn_cost: {:.4f}"
                        )
                        print(template.format(num, this_val_cost, c, avg_cost), flush=True, end='\r')
        except KeyboardInterrupt: #interrupted while training
            template = (
                " - KeyboardInterrupted at iter {}. final best model at iter {}. "
            )
            print(template.format(num, best_val_num))
            self.saver.restore(sess=self.sess, save_path=self.save_path)
        return self

    def predict(self, X):
        Y_pred = np.empty((0,2),int)
        for batch_X in batch_generator.tst_batch(X):
            self.saver.restore(sess=self.sess, save_path=self.save_path)
            batch_Y_pred = self.sess.run(self.predictor, feed_dict={self._X:batch_X, self._keeprate:1})
            Y_pred = np.append(Y_pred, batch_Y_pred, axis=0)
        y_pred = Y_pred[:,1]
        return y_pred

    def print_evaluation_result(self, oot_data_list):
        if param.segment == 'IC':
            top_k = 5000
        elif param.segment == 'MS':
            top_k = 15000

        for i, (X, y, cardNo) in enumerate(oot_data_list):
            y_hat = self.predict(X)
            recall_trans, recall_cards, top_k = metric.eval(y, y_hat, cardNo, top_k=top_k)
            template = (
                '{}: '
                '(recall_trans, recall_cards)='
                '({:.2f}, {:.2f}) (%, top_k={})'
            )
            print(template.format(param.oot_keys[i], recall_trans, recall_cards, top_k))

    def save_scores(self, oot_data_list):
        for i, (X, y, cardNo) in enumerate(oot_data_list):
            y_hat = self.predict(X)
            save_path = './prev_scores/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, self.model_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            def _save_score(card_list, y_list, yhat_list, yyyymm):
                with open('{}/{}_score.csv'.format(save_path, yyyymm), 'w') as sout:
                    sout.write('cardNo,y,score\n')
                    for card, y, yhat in zip(card_list, y_list, yhat_list):
                        sout.write ('%s,%s,%s\n'%(card, y, yhat))
            _save_score(cardNo, y, y_hat, param.oot_keys[i])

    def save_evaluation_result(self, oot_data_list, cutoff=None, top_k=10000):
        fname_record = 'record.csv'
        # record file existence check
        if not os.path.exists(fname_record):
            with open(fname_record, 'w') as fout:
                fout.write('model_id,')
                fout.write('batch_size,batch_fraud_ratio,minor_penalty,learning_rate,keeprate,model_name,model_size,')
                fout.write('cutoff(or top_k),')
                for i in range(len(oot_data_list)):
                    fout.write('{}_recall_tran,'.format(param.oot_keys[i]))
                    fout.write('{}_recall_card,'.format(param.oot_keys[i]))
                    fout.write('{}_top_k,'.format(param.oot_keys[i]))
                fout.write('\n')
        # record performance
        with open(fname_record, 'a') as fout:
            fout.write('%s,'%self.model_id)
            fout.write('%s,'%self.batch_size)
            fout.write('%s,'%self.batch_fraud_ratio)
            fout.write('%s,'%self.minor_penalty)
            fout.write('%s,'%self.learning_rate)
            fout.write('%s,'%self.keeprate)
            fout.write('%s,'%self.model_name)
            fout.write('%s,'%str(self.model_size).replace(',','-'))
            if cutoff:
                fout.write('%s,'%cutoff)
            else:
                fout.write('%s,'%top_k)
            for i, (X, y, cardNo) in enumerate(oot_data_list):
                y_hat = self.predict(X)
                recall_trans, recall_cards, top_k = metric.eval(y, y_hat, cardNo, cutoff=cutoff, top_k=top_k)
                fout.write('%s,%s,%s,'%(recall_trans, recall_cards, top_k))
            fout.write('\n')
    
    def export_weights(self):
        save_path = './weights/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, self.model_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with self.sess as sess:
            for v in tf.trainable_variables():
                print(v.name, end=' ')
                print(v.get_shape())
                fname = v.name.replace('/','-')
                fname = '{}.csv'.format(fname)
                fname = os.path.join(save_path, fname)
                np.savetxt(fname, v.eval(), delimiter=',')