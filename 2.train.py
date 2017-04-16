from itertools import product
import param
import dataio
import model
import metric
from wrapper import Model

if __name__ == '__main__':
    print('*** Load dataset...')
    trn_data_list, oot_data_list = dataio.get_datalist()

    learning_param_iterator = product(
        param.batch_size_list, 
        param.learning_rate_list, 
        param.keeprate_list
    )
    imbalance_param_iterator = zip(
        param.batch_fraud_ratio_list, 
        param.minor_penalty_list
    )
    for batch_size, learning_rate, keeprate in learning_param_iterator:
        for batch_fraud_ratio, minor_penalty in imbalance_param_iterator:
            num_iteration = param.num_model_per_config if param.TRAIN == 1 else 1
            for i in range(num_iteration):
                
                model_setting_list = []
                model_design, model_size, model_id = model.get_model_config(param.model_design)
                model_setting_list.append((model_design, model_size, model_id))

                for model_design, model_size, model_id in model_setting_list:
                    print('*** Define model...')
                    mdl = Model( 
                        model_design=model_design, 
                        model_size=model_size, 
                        model_id=model_id)

                    if param.TRAIN == 1:
                        print('*** Set learning parameter...')
                        mdl.parameter_setting(
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            keeprate=keeprate,
                            batch_fraud_ratio=batch_fraud_ratio,
                            minor_penalty=minor_penalty)

                        print('*** Train model...')
                        mdl.train(
                            trn_data_list, 
                            oot_monitoring=True,
                            oot_data_list=oot_data_list)

                    # final evaluation scores
                    if param.print_result is True:
                        print('*** Evaluation...')
                        mdl.print_evaluation_result(oot_data_list)
                    if param.save_individual_scores is True:
                        print('*** Save Scores...')
                        mdl.save_scores(oot_data_list)
                    if param.save_result is True:
                        print('*** Save Evaluation result...')
                        mdl.save_evaluation_result(oot_data_list, cutoff=0.70)
                        # mdl.save_evaluation_result(oot_data_list, top_k=15000)
                    if param.export_weight is True:
                        print('*** Export weights...')
                        mdl.export_weights()