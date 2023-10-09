import sys
sys.path.append('../')  # 将上一级目录添加到路径中
import parameters
import run_exp as run
from tqdm import tqdm
import gc
import ttab.configs.algorithms as agrth
args = parameters.get_args()

#设置job_id
# args.job_id='826_formal_1'
args.job_id='debug'

model_adaptation_method_choices=[
            # "no_adaptation",
            # "tent",
            # "bn_adapt",
            # # "memo",//too slow,multithread
            # "shot",
            # "t3a",
            # # "ttt",//too slow,multithread
            "note",
            # "sar",
            # "conjugate_pl",
            # "cotta",
            # "eata",
            ]

data_names_choice = [
    'gaussian_noise','snow','brightness','fog','frost','contrast','motion_blur','glass_blur','zoom_blur',
    'shot_noise','defocus_blur','elastic_transform',
    'jpeg_compression','pixelate','impulse_noise'
]
inter_domain_choices=[
            "HomogeneousNoMixture",
            "CrossMixture",
            "HeterogeneousNoMixture",
            # # "InOutMixture",
            # "CrossHeterMixture", 
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



# for domain in inter_domain_choices:
    
    
#     args.batch_size=100
#     args.non_iid_ness=0.01
#     args.inter_domain = domain
    
#     for adaption_method in model_adaptation_method_choices:
#         args.model_adaptation_method = adaption_method
#         i = 3 
#         j=0
#         corrruption = []
#         while j != i :
#             corrruption.append('cifar10_c_deterministic-' + data_names_choice[j] + '-5')
#             if (j + 1) % 3 == 0:
#                 args.data_names = ';'.join(corrruption)
#                 # print(domain, adaption_method, args.data_names)
                
#                 args.corruption_num= j+1    
#                 run.main(init_config=args)
#                 if i == 15:
#                     break
#                 else:
#                     i = i + 3
                
#                 gc.collect()
#             j=j+1

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

args = parameters.get_args()
args.job_id='817_formal_2'
non_iid_ness_choices = [0.01, 0.03, 0.09, 0.27, 0.81, 1]                   
args.batch_size=100
args.inter_domain = "CrossHeterMixture"                

for non_iid in tqdm(non_iid_ness_choices):
    args.non_iid_ness = non_iid
    
    for adaption_method in tqdm(model_adaptation_method_choices):
        if adaption_method=='note':
            args.ckpt_path='/data/TTA-exp/TTAB-jiang/TTAB/ttab-main/ttab-benchmarkData/rn26_bn.pth'
            print('!!!!!!!!!!!',args.ckpt_path)
            agrth.algorithm_defaults['note']['iabn']="True"
        else:
            agrth.algorithm_defaults['note']['iabn']="False"
            args.ckpt_path='TTAB-jiang/TTAB/ttab-main/ttab-benchmarkData/rn26_bn.pth'
            print('!!!!!!!!!!!',args.ckpt_path)
        args.model_adaptation_method = adaption_method
        i = 3
        corrruption = []
        j=0
        while j != i :
            corrruption.append('cifar10_c_deterministic-' + data_names_choice[j] + '-5')
            
            if (j + 1) % 3 == 0:
                data_name = ';'.join(corrruption)
                args.data_names = data_name
                # print(domain, adaption_method, data_name)
                
                args.corruption_num= j+1    
                run.main(init_config=args)
                if i == 15:
                    break
                else:
                    i = i + 3
                corrruption = []
                gc.collect()
            j=j+1
            



