import pandas as pd
import numpy as np
import os
import csv
import ast
import re
import matplotlib.pyplot as plt
import pickle
import threading
from tqdm import tqdm

# base imformation
model_adaptation_method_choices=[
            "no_adaptation",
            "tent",
            # "bn_adapt",
            # "memo",
            # "shot",
            # "t3a",
            # "ttt",
            "note",
            "sar",
            # "conjugate_pl",
            "cotta",
            # "eata",
            ]

data_names_choice = [
    'snow','brightness','fog','frost','contrast','motion_blur','glass_blur','zoom_blur',
    'gaussian_noise','shot_noise','defocus_blur','elastic_transform',
    'jpeg_compression','pixelate','impulse_noise'
]
data_choice = ['3', '6', '9', '12', '15']
non_iid_ness_choices = [0.01, 0.03, 0.09, 0.27, 0.81, 1] 
inter_domain_choices=[
            "HomogeneousNoMixture",
            "CrossMixture",
            "HeterogeneousNoMixture",
            # "InOutMixture",
            "CrossHeterMixture",
        ]
models=['resnet26',]
datasets=['cifar10',]




def read_file_lines(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield line
def process_gradientRawData(job_id):
    
    GradientL2_csv_output_path=grad_save_path+"/"+job_id+'/'+'loss_grad.csv'
    column = ['adaptation','inter_domain','data_num','step','loss','l2_weight']
    # pattern = re.compile(r".*_2")
    os.makedirs(grad_save_path+"/"+job_id,exist_ok=True)
    flag=0
    with open(GradientL2_csv_output_path, 'w', newline='') as csv_file:
              
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column)         

        for dirpath, dirnames, filenames in os.walk(gradient_path):
            # print('in')  d
            # print(dirpath)
            for file in filenames:
                  
                filename = os.path.basename(dirpath)
                filename = filename.split('_')
                file_jobname = '_'.join(filename[:3])
                if file_jobname == job_id:
                    grass = os.path.join(dirpath, file)
                    # print(grass)
                    with open(grass, 'rb') as file:

                        data = pickle.load(file)
                        append=[filename[5],filename[6],filename[7],data['step'], data['loss']]
                        # print(data)
                        # for i in data['bn1.weight'].cpu().numpy():
                        #     append.append(i)
                        weight=[]
                        for key in data.keys():
                            if 'weight'in key:
                                weight.append(data[key].cpu().numpy()[0])
                        combined_data = np.concatenate(weight, axis=None)

                        l2_weight = np.linalg.norm(combined_data, ord=2)
                        append.append(l2_weight)
                        csv_writer.writerow(append)
                else:
                    # print('unmatch')
                    pass
                    

            # print(dirpath+"\ndone")




def process_AccuracyRawData(RawDataPath, SavePath, adaption, domain_noniid, data_nums):
    pattern2 = re.compile(r".*_2")

    csv_output_path = SavePath + '/output.csv'
    file_path = RawDataPath
    matching_lines = []
    pattern = r"'cross_entropy': (.*?), 'accuracy_top1': (.*?)\."
    
    for line in read_file_lines(file_path):
        match = re.search(pattern, line)
        if match:
            cross_entropy = match.group(1)
            accuracy_top1 = match.group(2)
            # print(RawDataPath)
            matching_lines.append((cross_entropy, accuracy_top1))
            # print(cross_entropy, accuracy_top1)
    if not os.path.exists(csv_output_path):
        with open(csv_output_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            #title
            
            if pattern2.match(job_id):
                # print('re job_id1',pattern2.match(job_id))
                csv_writer.writerow(["inter_domain","non_iid","adaption","data_num","cross_entropy", "accuracy"])        
            else:
                # print('re job_id2',pattern2.match(job_id))
                csv_writer.writerow(["inter_domain","non_iid","adaption","data_num","cross_entropy", "accuracy"])
    with open(csv_output_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if pattern2.match(job_id):
            non_iid_index = 0  # Index for non_iid_ness_choices
            for cross_entropy, accuracy_top1 in matching_lines:
                non_iid = non_iid_ness_choices[non_iid_index]  # Get the non_iid value
                csv_writer.writerow([domain_noniid,non_iid, adaption, data_nums, cross_entropy, accuracy_top1])
                non_iid_index += 1
        else:
            non_iid = 0.01
            for cross_entropy, accuracy_top1 in matching_lines:
                csv_writer.writerow([domain_noniid, non_iid , adaption, data_nums, cross_entropy, accuracy_top1])


#"accuracy analysis"
# raw data path
accuracy_path = '/data/TTA-exp/TTAB-jiang/TTAB/ttab-main/data/logs/'
# result path
accuracy_save_path='/data/TTA-exp/ttab-analysis_result/accuracy'
#"gradient analysis"
gradient_path = '/data/TTA-exp/TTAB-jiang/TTAB/ttab-main/data/logs/resnet26/cifar10/loss_grad'
grad_save_path='/data/TTA-exp/ttab-analysis_result/accuracy'
grad_save_path='/data/TTA-exp/ttab-analysis_result/accuracy'




def process_AccuracyData(job_id):
    job_name = job_id
    AccuracySavePath = accuracy_save_path +'/'+ job_name
    pattern2 = re.compile(r".*_2")
    
    if pattern2.match(job_id):
        inter_domain_choices=["CrossHeterMixture",]
    else:
        inter_domain_choices=["HomogeneousNoMixture","CrossMixture","HeterogeneousNoMixture", "CrossHeterMixture",]
    
    os.makedirs(AccuracySavePath,exist_ok=True)
    
    for model in models:
        for dataset in datasets:
            for method in model_adaptation_method_choices:
                for domain in inter_domain_choices:

                    for num in data_choice:

                        datapath=""
                        dataPath = accuracy_path +\
    f'{model}/{dataset}'+f'/{job_name}_{model}_{dataset}_{method}_{domain}_{num}/log.txt'
                        process_AccuracyRawData(dataPath, AccuracySavePath, method, domain, num)
                        # print(dataPath)
    
    
def process_AccuracyData_thread(job_id):
    process_AccuracyData(job_id)
    
    print(f'{job_id}accuracy done')

def process_gradientRawData_thread(job_id):
    process_gradientRawData(job_id)
    print(f'{job_id}gradient done')

if __name__ == '__main__':
    job_ids = ['104_formal_1', '104_formal_2']  # 添加更多的job_id，以便同时运行多个任务

    threads = []
    for job_id in job_ids:
        thread = threading.Thread(target=process_AccuracyData_thread, args=(job_id,))
        threads.append(thread)
        gradient_thread = threading.Thread(target=process_gradientRawData_thread, args=(job_id,))
        threads.append(gradient_thread)

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in tqdm(threads):
        thread.join()

def main(job_ids):
    # job_ids = ['921_formal_1', '921_formal_2']  # 添加更多的job_id，以便同时运行多个任务
    

    threads = []
    for job_id in job_ids:
        thread = threading.Thread(target=process_AccuracyData_thread, args=(job_id,))
        threads.append(thread)
        gradient_thread = threading.Thread(target=process_gradientRawData_thread, args=(job_id,))
        threads.append(gradient_thread)

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in tqdm(threads):
        thread.join()        
