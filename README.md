



先将AutoMaster_TrainSet 和 AutoMaster_TestSet 拷贝到data 路径下 再使用 .



代码结构

+ notebook 课件   
    ....
+ result 结果保存路径
    ....    
+ seq2seq_tf2 模型结构
    ....
+ utils 工具包
    + config  配置文件
    + data_loader 数据处理模块
    + multi_proc_utils 多进程数据处理
+ data  数据集
    + AutoMaster_TrainSet 拷贝数据集到该路径
    + AutoMaster_TestSet  拷贝数据集到该路径
    ....
    
    
训练步骤:
1. 拷贝数据集到data路径下
2. 运行utils\data_loader.py可以一键完成 预处理数据 构建数据集
3. 训练模型 运行seq2seq_tf2\train .py脚本 或者 05_seq2seq-Train.ipynb 均可以完成训练
4. 测试 参照05_seq2seq-Train.ipynb中有现成代码,包括结果生成
5. 结果提交,参考5_2研讨课提交流程,线上提交验证结果.

* 05_seq2seq-Train.ipynb 线上得分27.8分 (epochs 5)
* beam search 代码在seq2seq_tf2\test_helper.py中 ,自行参考实现.
* GPU运行速度参考, embedding_dim_300 enc_max_len=200 units=512 500s/epochs (8W训练集) 
    
提交结果:

1. score 19.3103

> score 19.3103 gru 1024 ,embedding_dim 300 batch_size=16 Epoch=10 训练时间2700s,去除标点符号

2. score 22.537

> loss 1.03  

> batch_size_64_epochs_10_max_length_inp_299_embedding_dim_300

3. score 23.1084

skip gram + learning rate epochs_4 1e-5 -> epochs_1 1e-4

> 2019_12_06_18_19_33_batch_size_64_epochs_4_max_length_inp_299_embedding_dim_300.csv


4. score 27.9942   

> loss 1.0

learning rate 1e-4 优化数据预处理 添加标点 去除`[]`,优化切词

> 2019_12_07_12_10_34_batch_size_32_epochs_4_max_length_inp_200_embedding_dim_300


5. score 28.4136

> 2019_12_07_15_43_36_batch_size_32_epochs_10_max_length_inp_200_embedding_dim_300_4_1_submit_proc_add_masks_loss.csv


6. score 27.965

> 2019_12_07_17_56_34_batch_size_32_epochs_10_max_length_inp_200_embedding_dim_500_4_1_submit_proc_add_masks_loss.csv


7. pgn + coverage + coverage_loss + 部分 mask

log_loss+0.5*coverage_loss:


8. pgn + coverage + coverage_loss
epochs=5
```
{'rouge-1': {'f': 0.29264068869911036,
             'p': 0.4435294117647059,
             'r': 0.29155785391079514},
 'rouge-2': {'f': 0.058585856857769666,
             'p': 0.09, 
             'r': 0.04395604395604395},
 'rouge-l': {'f': 0.23421453925590519,
             'p': 0.4435294117647059,
             'r': 0.29155785391079514}}
```


training_checkpoints_pgn_cov
epochs=28 
```
{'rouge-1': {'f': 0.4696623331991586,
             'p': 0.5895238095238095,
             'r': 0.4281344386607545},
 'rouge-2': {'f': 0.2069264042808606,
             'p': 0.21428571428571427,
             'r': 0.2542857142857143},
 'rouge-l': {'f': 0.3854431036362561,
             'p': 0.535952380952381,
             'r': 0.3984641089904247}}
```