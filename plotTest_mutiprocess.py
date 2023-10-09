

import parameters
import run_exp as run
from tqdm import tqdm
import gc
import threading
import ttab.configs.algorithms as agrth
#mission_id = '9263'
mission_id = '104_formal_'


model_adaptation_method_choices=[
            "no_adaptation",
            "tent",
            # "bn_adapt",
            # "memo",//too slow,multithread
            # "shot",
            # "t3a",
            # "ttt",//too slow,multithread
            "note",
            "sar",
            # # "conjugat`e_pl",
            "cotta",
            # # "eata",
            ]
data_names_choice = [
    'snow','brightness','fog','frost','contrast','motion_blur','glass_blur','zoom_blur',
    'gaussian_noise','shot_noise','defocus_blur','elastic_transform',
    'jpeg_compression','pixelate','impulse_noise'
]
inter_domain_choices=[
            "HomogeneousNoMixture",
            "CrossMixture",
            "HeterogeneousNoMixture",
            # # "InOutMixture",
            "CrossHeterMixture",
        ]




# experiment 1
'''
第一组：四种环境进行比较：

  环境1： "--inter_domain" 参数设置为 "HomogeneousNoMixture"
         "--batch_size" 参数设置为 100
         上述两个参数固定后，更改以下两个参数：
         "--model_adaptation_method" 将choices中的参数分别都设置一次（在每一个方法上查看结果）
         在"--model_adaptation_method"固定后，

	更改 "--data_names"参数，
         "--data_names"中一共有15个类型的corruption,目前在parameters.py里我一共写了六个，其余的可以对应数据集一一查找。

         "--data_names"参数更改的方法：每次添加三种corruption进去，直到所有corruption添加完毕；
         例如：此处用1-15的编号分别代表15种类型的corruption，第一次为（1，2，3），第二次为（1，2，3，4，5，6），第三次为
         （1，2，3，4，5，6，7，8，9）....直到第五次为（1，2，3，4，..., 15),这里corruption加进去的顺序可以随机挑选，但是此处一旦确定，在跑环境2,3,4时要与此处保持一致。


  环境2："--inter_domain" 参数设置为 "CrossMixture"
         "--batch_size" 参数设置为 100
         其余两个参数设定与环境1保持一致
  环境3："--inter_domain" 参数设置为 "HeterogeneousNoMixture"
         "--batch_size" 参数设置为 100
         "--non_iid_ness" 参数设置为0.01
         其余两个参数设定与环境1保持一致
  环境4："--inter_domain" 参数设置为 "CrossHeterMixture"
         "--batch_size" 参数设置为 100
         "--non_iid_ness" 参数设置为0.01
         其余两个参数设定与环境1保持一致
'''


def exp1(inter_domain_choices):
    print("Starting exp1...")
    
    for domain in tqdm(inter_domain_choices):
        
        
        for adaption_method in tqdm(model_adaptation_method_choices):
            i = 3
            corrruption = []
            j=0
            while j != i :
                corrruption.append('cifar10_c_deterministic-' + data_names_choice[j] + '-5')
                if (j + 1) % 3 == 0:
                    args = parameters.get_args()
                    if adaption_method=='note':
                        args.ckpt_path='/data/TTA-exp/TTAB-jiang/TTAB/ttab-main/ttab-benchmarkData/rn26_iabn.pth'
                    else: 
                        args.ckpt_path='/data/TTA-exp/TTAB-jiang/TTAB/ttab-main/ttab-benchmarkData/rn26_bn.pth'
                    args.job_id = mission_id + str(1)
                    args.inter_domain = domain
                    args.model_adaptation_method = adaption_method
                    data_name = ';'.join(corrruption)
                    args.device="cuda:0"
                    args.batch_size=64
                    args.non_iid_ness=0.01
                    args.data_names = data_name
                    args.corruption_num= j+1    
                    print(f"Running experiment for domain: {domain},\nadaptation method: {adaption_method},\ndata names: {data_name},\nmodel_path:{args.ckpt_path}\n")
                    print()
                    
                    run.main(init_config=args)
                    
                    gc.collect()
                    # if i == 6:
                    if i == 15:
                        break
                    else:
                        i = i + 3
                    
                j=j+1
    # pass
    print("exp1 has finished.")

'''
第二组实验 "--inter_domain" 参数设置为 "CrossHeterMixture"
         "--batch_size" 参数设置为 100
         "--non_iid_ness" 参数分别设置为0.01, 0.03, 0.09, 0.27, 0.81, 1每设置一次都要进行以下重复实验：
         更改以下两个参数：
         "--model_adaptation_method" 将choices中的参数分别都设置一次（在每一个方法上查看结果）
         在"--model_adaptation_method"固定后，更改 "--data_names"参数，
         "--data_names"中一共有15个类型的corruption,目前在parameters.py里我一共写了六个，其余的可以对应数据集一一查找。
         "--data_names"参数更改的方法：每次添加三种corruption进去，直到所有corruption添加完毕；
         例如：此处用1-15的编号分别代表15种类型的corruption，第一次为（1，2，3），第二次为（1，2，3，4，5，6），第三次为
         （1，2，3，4，5，6，7，8，9）....直到第五次为（1，2，3，4，..., 15),这里corruption加进去的顺序可以随机挑选，但是此处
         一旦确定，在重复实验时需要和第一次一致。

'''

                  
                
def exp2(non_iid_ness_choices):
    
    print("Starting exp2...")
    for non_iid in tqdm(non_iid_ness_choices):  
        for adaption_method in tqdm(model_adaptation_method_choices):
            i = 3
            corrruption = []
            j=0
            while j != i :
                corrruption.append('cifar10_c_deterministic-' + data_names_choice[j] + '-5')
                if (j + 1) % 3 == 0:
                    args = parameters.get_args()
                    args.job_id = mission_id+str(2)
                    args.non_iid_ness = non_iid
                    if adaption_method=='note':
                        args.ckpt_path='/data/TTA-exp/TTAB-jiang/TTAB/ttab-main/ttab-benchmarkData/rn26_iabn.pth'
                    else:
                        args.ckpt_path='/data/TTA-exp/TTAB-jiang/TTAB/ttab-main/ttab-benchmarkData/rn26_bn.pth'
                    args.model_adaptation_method = adaption_method
                    args.device="cuda:1"
                    args.batch_size=100
                    args.inter_domain = "CrossHeterMixture"
                    data_name = ';'.join(corrruption)
                    args.data_names = data_name
                    # print(domain, adaption_method, data_name)
                    args.corruption_num= j+1    
                    print(f"Running experiment for domain: {args.inter_domain} in {non_iid},\nadaptation method: {adaption_method},\ndata names: {data_name},\niabn:{agrth.algorithm_defaults['note']['iabn']},\nmodel_path:{args.ckpt_path}\n")
                    run.main(init_config=args)
                    gc.collect()
                    
                    if i == 15:
                        break
                    else:
                        i = i + 3
                    

                j=j+1
    # pass
    print("exp2 has finished.")



#设置job_id
# args1.job_id = 'debug'

t1 = threading.Thread(target=exp1,args=(inter_domain_choices,))
t1.start()
# exp1(inter_domain_choices)


# args2.job_id = 'debug'
        
non_iid_ness_choices = [0.01, 0.03, 0.09, 0.27, 0.81, 1] 

t2 = threading.Thread(target=exp2,args=(non_iid_ness_choices,))
t2.start()
t2.join()
# exp2(non_iid_ness_choices)

print('done')

