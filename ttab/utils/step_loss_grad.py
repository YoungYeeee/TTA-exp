import os
from typing import Any
import pickle
import torch
import pandas as pd
import numpy as np
import csv
DomainNum=np.zeros(16)
Combine=None
def save(conf: Any,state):
    pass
        # conf.grad_dir = os.path.join(
        #     # #可修改为其他路径
        #     # "E:/GITHUB/TTA-Exp/ttab/data/step_loss_grad",
        #     # conf.model_name,
        #     # conf.job_name,
        #     # f"{conf.step}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}"
        #     conf.root_path,
        #     conf.model_name,
        #     conf.base_data_name,
        #     'loss_grad',
        #     # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{int(conf.timestamp if conf.timestamp is not None else time.time())}-seed{conf.seed}",
        #     # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{str(time.time()).replace('.', '_')}-seed{conf.seed}",
        #     f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}_{conf.corruption_num}",
        #     # f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}",
        #     )
        # if not os.path.exists(conf.grad_dir):
        #     os.makedirs(conf.grad_dir)
        # temp=state
        # loss = state["loss"]
        # # if conf.step in None:
        # #      print('conf.step is none')
        # # elif temp['loss'] is None:
        # #      print('state loss is none')
        # # elif temp['grads'] is None:
        # #      print('state grads is none')
        # # print("the loss is ", loss)
        # # print(conf)
        # dict={
        #      "step":conf.step,
        #      "loss":temp["loss"],
        #      **temp["grads"]
        #       }
        # pickle_file = os.path.join(conf.grad_dir, f'{conf.step}_save.pickle')
        # # print(f"file path:{pickle_file}")
        # with open(pickle_file, 'ab') as f:
        #     pickle.dump(dict, f)


def saveAsCSV(conf: Any , state , batch):
    conf.grad_dir = os.path.join(
            # 可修改为其他路径
            # "E:/GITHUB/TTA-Exp/ttab/data/step_loss_grad",
            # conf.model_name,
            # conf.job_name,
            # f"{conf.step}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}"
            conf.root_path,
            conf.model_name,
            conf.base_data_name,
            'csvData',
            # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{int(conf.timestamp if conf.timestamp is not None else time.time())}-seed{conf.seed}",
            # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{str(time.time()).replace('.', '_')}-seed{conf.seed}",
            # f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}_{conf.corruption_num}",
            # f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}",
            )
    if not os.path.exists(conf.grad_dir):
        os.makedirs(conf.grad_dir)
    print(conf)
    if state["grads"] is not None:
        L2=getL2(state["grads"])
    else:
        L2=0
    Accuracy_Top1=accuracy_top1(
        target=batch._y, 
        output=state["yhat"]
        )
    Cross_Entropy=cross_entropy(    
        output=state["yhat"], 
        target=batch._y
        )
    # add
    global DomainNum,Combine
    headers = ["step", 
               "loss", 
               "L2", 
               "Accuracy_Top1", 
               "Cross_Entropy", 
               "model_adaptation_method", 
               "dataset", 
               "batch_size", 
               #"data_names",
               "inter_domain",
               "model_name",
               "Combine"
               ]
    dict={
             "step":conf.step,
             "loss":state["loss"],
             "L2":L2,
             "Accuracy_Top1":Accuracy_Top1,
             "Cross_Entropy":Cross_Entropy,
             "model_adaptation_method":conf.model_adaptation_method,
             "dataset":conf.base_data_name,
             "batch_size":conf.batch_size,
             #"data_names":conf.data_names,
             "inter_domain":conf.inter_domain,
             "model_name":conf.model_name,
             "Combine":Combine,
        }
    corruptions = conf.data_names.split(';')
    #dict.update({"domain_length":corruptions.size()})
    #names=[]
    i=1
    for corr in corruptions:
        name = corr.split('-')[-2]
        headers.append(name)
        dict.update({f'{name}':DomainNum[i]})
        #dict.append(corr)
        #dict[corr]=DomainNum[i]
        i=i+1
        #names.append(corr)   
    # add
    #+ list(temp["grads"].keys())
    DomainNum = np.zeros(16)
    Combine = None
    csv_file = os.path.join(conf.grad_dir, f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}_{conf.corruption_num}.csv")
  
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)  
            writer.writeheader()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(dict)
        


def getL2(data):
    weight=[]
    for key in data.keys():
        if 'weight'in key:
            weight.append(data[key].cpu().numpy()[0])
    combined_data = np.concatenate(weight, axis=None)
    #L2范式（保存每一步的全部梯度的L2范式）
    l2_weight = np.linalg.norm(combined_data, ord=2)
    return l2_weight

# ACC
def _accuracy(target, output, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size).item()

def accuracy_top1(target, output, topk=1):
    """Computes the precision@k for the specified values of k"""
    return _accuracy(target, output, topk)

# CrossEntropy
cross_entropy_loss = torch.nn.CrossEntropyLoss()

def cross_entropy(target, output):
    """Cross entropy loss"""
    return cross_entropy_loss(output, target).item()