import os
import pickle
import numpy as np
from random import random
from data_loader import get_loader
from solver import Solver
import torch
# ddddd
import os
import argparse
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
import optuna
import shap  # 确保已安装 SHAP：pip install shap
import optuna.visualization as vis
from optuna.visualization import plot_parallel_coordinate
word_emb_path = '../glove.840B.300d.txt'
assert(word_emb_path is not None)


username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY'), 'cmdc': data_dir.joinpath('CMDC'),
             'iemocap': os.path.join(data_dir,'IEMOCAP')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:       #此时kwargs是字典
            for key, value in kwargs.items(): #kwargs.items()生成键值对(key, value) 
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)  #self是对象

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[self.data.lower()]
        self.sdk_dir = sdk_dir
        # Glove path
        self.word_emb_path = word_emb_path
        self.output_dim = 1 # 回归任务通常输出 1 维（如情感分数、数值预测等）

        # Data Split ex) 'train', 'valid', 'test'
        # self.data_dir = self.dataset_dir.joinpath(self.mode)
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()  #创建一个 ArgumentParser 对象
    parser.add_argument('--best_model_Configuration_Log', type=str, default='./src/best_Configuration_optuna625.txt',
                        help='Load the best model to save features')
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)

    # Bert
    parser.add_argument('--use_bert', type=str2bool, default=True)
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=10)

    # weight
    parser.add_argument('--diff_weight', type=float, default=0.3)
    parser.add_argument('--sim_weight', type=float, default=0.6)
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=0.5)

    #moe的负载均衡
    parser.add_argument('--aux_loss_weight', type=float, default=0.01)
    parser.add_argument('--rank_weight', type=float, default=0.01)
    parser.add_argument('--boost_weight', type=float, default=0.1)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='relu')

    # Model
    parser.add_argument('--model', type=str,
                        default='MISA', help='one of {MISA, MISA_CMDC}')

    # Data
    parser.add_argument('--data', type=str, default='mosi')  # cmdc\mosi\iemocap\mosei------------------------------------
    # cmdc的5折交叉验证
    parser.add_argument("--cross_validation", type=str,
                        choices=["cmdc_data_all_modal_1", "cmdc_data_all_modal_2", "cmdc_data_all_modal_3",
                                 "cmdc_data_all_modal_4", "cmdc_data_all_modal_5"], default="cmdc_data_all_modal_1")

    # Parse arguments 解析命令行参数
    if parse:
        kwargs = parser.parse_args() #kwargs存储命令行参数 name+default
    else:
        kwargs = parser.parse_known_args()[0]

    # print(kwargs.data)
    if kwargs.data == "mosi":
        kwargs.num_classes = 1
        kwargs.batch_size = 64
    elif kwargs.data == "mosei":
        kwargs.num_classes = 1
        kwargs.batch_size = 16
    elif kwargs.data == "ur_funny":
        kwargs.num_classes = 2
        kwargs.batch_size = 32
    elif kwargs.data == "iemocap":
        kwargs.num_classes = 8
        kwargs.batch_size = 32
        kwargs.patience = 15
        kwargs.model = 'MISA_CMDC'
    elif kwargs.data == "cmdc":
        kwargs.num_classes = 1
        kwargs.batch_size = 6
        kwargs.use_bert = False
        kwargs.model = 'MISA_CMDC'
        kwargs.embedding_size = 768
        kwargs.cross_validation = "cmdc_data_all_modal_1"  # 修改进行交叉验证： 1 2 3 4 5
    else:
        print("No dataset mentioned")
        exit()

    # Namespace => Dictionary
    kwargs = vars(kwargs)  #kwargs字典 
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


def reset_seed(seed=336):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义目标函数
def objective(trial):
    # reset_seed1 = trial.suggest_int("reset_seed1", 1, 1000)
    # reset_seed(reset_seed1)  # 必不可少,否则复现不出来
    # reset_seed(336)  # 必不可少,否则复现不出来  473
    reset_seed(473)
    # 定义需要搜索的超参数及范围
    sim_weight = trial.suggest_categorical("sim_weight", [1.0, 0.8, 0.6, 0.5, 0.3, 0.1])  # 整数范围
    diff_weight = trial.suggest_categorical("diff_weight", [1.0, 0.5, 0.3, 0.1, 0.05])  # 整数范围
    recon_weight = trial.suggest_categorical("recon_weight", [5.0, 3.0, 1.0, 0.5, 0.3, 0.1])  # 整数范围
    learning_rate = trial.suggest_categorical("learning_rate", [5e-4, 1e-4, 5e-5, 1e-5])  # 离散集合
    # aux_loss_weight = trial.suggest_categorical("aux_loss_weight", [0.1, 0.05, 0.01, 0.005])  # 离散集合
    # rank_weight = trial.suggest_categorical("rank_weight", [0.1, 0.05, 0.01, 0.005])  # 离散集合
    # boost_weight= trial.suggest_categorical("boost_weight", [0.1, 0.05, 0.01, 0.005])  # 离散集合
    aux_loss_weight = trial.suggest_categorical("aux_loss_weight", [0.15, 0.1, 0.05, 0.01, 0.005])  # 离散集合
    rank_weight = trial.suggest_categorical("rank_weight", [0.1, 0.05, 0.01, 0.005])  # 离散集合
    boost_weight= trial.suggest_categorical("boost_weight", [0.15, 0.1, 0.05, 0.01])  # 离散集合
    # shifting_weight = trial.suggest_float("shifting_weight", 1.0, 3.0) #连续范围      ------------------------------
    # order_center_weight = trial.suggest_categorical("order_center_weight", [2.0, 1.0, 0.5])  # 离散集合
    # order_center_weight = trial.suggest_float("order_center_weight", 0.5, 2)
    # ce_loss_weight = trial.suggest_float("ce_loss_weight", 1, 3)  # 整数范围
    # optimizer_c = trial.suggest_categorical("optimizer_c", ['RMSprop', 'Adam'])  # 离散集合
    # center_score_weight = trial.suggest_float("center_score_weight", 0.00, 0.1)  # 连续范围     ---------------------------
    activation = trial.suggest_categorical("activation", ["leakyrelu", "prelu", "relu", "rrelu", "tanh"])  # 离散集合
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"

    # Setting the config for each stage
    train_config = get_config(mode='train')
    train_config.sim_weight=sim_weight
    train_config.diff_weight = diff_weight
    train_config.recon_weight = recon_weight
    train_config.learning_rate = learning_rate
    train_config.aux_loss_weight = aux_loss_weight
    train_config.rank_weight = rank_weight
    train_config.boost_weight = boost_weight
    # train_config.order_center_weight = order_center_weight
    # train_config.ce_loss_weight = ce_loss_weight
    # train_config.pred_center_score_weight = center_score_weight
    train_config.activation = activation_dict[activation]

    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    # Creating pytorch dataloaders  批量加载数据
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)

    # Solver is a wrapper for model traiing and testing
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    MAE = solver.train()

    return MAE  # 返回目标值



if __name__ == '__main__':
     
    # Setting random seed  伪随机，方便复用
    random_name = str(random())
    random_seed = 336   
    torch.manual_seed(random_seed)      #为CPU随机生成器设定种子
    torch.cuda.manual_seed_all(random_seed)  #为GPU
    torch.backends.cudnn.deterministic = True  #确定性算法
    torch.backends.cudnn.benchmark = False   #性能调优
    np.random.seed(random_seed)   #为 NumPy 库中的随机数生成器设定种子

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定显卡使用第4卡，（0是第一块）


    study_name = "optimization_study"
    storage = f"sqlite:///{study_name}.db"
    # 开始超参数优化
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, sampler=optuna.samplers.TPESampler(seed=336), load_if_exists=True)
    # 如果已经完成优化，直接加载，无需重新优化
    if len(study.trials) == 0:
        # 定义固定参数
        fixed_params = {
            "sim_weight": 0.80,
            "diff_weight": 0.5,
            "recon_weight": 5.0,
            "learning_rate": 5e-5,
            "aux_loss_weight": 0.05,
            "rank_weight": 0.05,
            "boost_weight": 0.05,
            # "order_center_weight": 1.0,
            # "ce_loss_weight": 1.0,
            # "center_score_weight": 0.0,
            "activation": "rrelu",

        }

        # 手动插入固定参数为一个试验
        study.enqueue_trial(fixed_params)

        study.optimize(objective, n_trials=200)

        # 打印最佳参数
        print("Best parameters:", study.best_params)
        print("Best MAE:", study.best_value)

        # 可视化优化结果
        vis.plot_optimization_history(study).show()
        vis.plot_parallel_coordinate(study).show()
    else:
        print("Loaded existing Study from database.")

        # 加载 Study
        loaded_study = optuna.load_study(study_name="optimization_study", storage=f"sqlite:///{study_name}.db")

        # 输出最优参数与结果
        print("Best hyperparameters:", loaded_study.best_trial.params)
        print("Best value (objective):", loaded_study.best_trial.value)

        # 用最优参数再运行一次
        best_params = loaded_study.best_trial.params
        result = objective(optuna.trial.FixedTrial(best_params))
        print("Re-evaluated result with best params:", result)

        # 自定义并行坐标图的线条粗细
        fig = plot_parallel_coordinate(loaded_study)

        # 修改线条宽度
        for trace in fig.data:
            if trace.type == 'scatter':
                trace.line.width = 5  # 修改线条宽度为 2

        # 显示图
        fig.show()


# 展示的命令
# pip install optuna-dashboard
# optuna-dashboard sqlite:///optimization_study.db