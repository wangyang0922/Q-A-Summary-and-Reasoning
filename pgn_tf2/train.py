# -*- coding:utf-8 -*-
# Created by LuoJie at 11/29/19

from utils.gpu_utils import config_gpu

import tensorflow as tf

from pgn_tf2.batcher import batcher
from pgn_tf2.model import PGN
from pgn_tf2.train_helper import train_model
from utils.params_utils import get_params
from utils.wv_loader import Vocab
import numpy as np


def train(params):
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    print("Building the model ...")
    vocab = Vocab(params["vocab_path"], params["max_vocab_size"])
    params['vocab_size'] = vocab.count

    # 构建模型
    print("Building the model ...")
    # model = Seq2Seq(params)
    model = PGN(params)

    print("Creating the batcher ...")
    dataset, params['steps_per_epoch'] = batcher(vocab, params)

    # print('dataset is ', dataset)

    # 获取保存管理者
    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, params['checkpoint_dir'], max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
        params["trained_epoch"] = int(checkpoint_manager.latest_checkpoint[-1])
    else:
        print("Initializing from scratch.")
        params["trained_epoch"] = 1

    # 学习率衰减
    params["learning_rate"] *= np.power(0.95, params["trained_epoch"])
    print('learning_rate:{}'.format(params["learning_rate"]))
    # 训练模型
    print("Starting the training ...")

    train_model(model, dataset, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数m
    params = get_params()
    params['pointer_gen'] = True
    params['use_coverage'] = True
    params['enc_units'] = 256
    params['dec_units'] = 512
    params['max_enc_len'] = 250
    params['max_dec_len'] = 45
    params['batch_size'] = 16
    params['attn_units'] = 20
    params['eps'] = 1e-12

    # 训练模型
    train(params)
