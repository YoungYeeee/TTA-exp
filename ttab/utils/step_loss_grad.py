import os
from typing import Any
import pickle


def save(conf: Any,state):
        conf.grad_dir = os.path.join(
            # #可修改为其他路径
            # "E:/GITHUB/TTA-Exp/ttab/data/step_loss_grad",
            # conf.model_name,
            # conf.job_name,
            # f"{conf.step}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}"
            conf.root_path,
            conf.model_name,
            conf.base_data_name,
            'loss_grad',
            # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{int(conf.timestamp if conf.timestamp is not None else time.time())}-seed{conf.seed}",
            # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{str(time.time()).replace('.', '_')}-seed{conf.seed}",
            f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}_{conf.corruption_num}",
            # f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}",
            )
        if not os.path.exists(conf.grad_dir):
            os.makedirs(conf.grad_dir)
        temp=state
        loss = state["loss"]
        # if conf.step in None:
        #      print('conf.step is none')
        # elif temp['loss'] is None:
        #      print('state loss is none')
        # elif temp['grads'] is None:
        #      print('state grads is none')
        # print("the loss is ", loss)
        print(conf)
        dict={
             "step":conf.step,
             "loss":temp["loss"],
             **temp["grads"]
              }
        pickle_file = os.path.join(conf.grad_dir, f'{conf.step}_save.pickle')
        # print(f"file path:{pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump(dict, f)