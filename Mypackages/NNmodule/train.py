# _*_ coding:utf-8 _*_  
import os
import tensorflow as tf 
import numpy as np 
from .model import Network
from ..custom_tools import Config_Tools


class Train:
    r"""训练类型，提供训练函数
    """
    def __init__(self):
        r"""初始化配置，载入numpy训练集，以及验证集
        """
        self.__cfgtool = Config_Tools()
        self.__cfg = self.__cfgtool.cfg_to_dict('NN')
        self.CKPT_DIR = self.__cfg['cpkt_dir']
        self.train_data = np.load(self.__cfg['train_data'])
        self.train_label = np.load(self.__cfg['train_label'])
        self.verification_data = np.load(self.__cfg['verification_data'])
        self.verification_label = np.load(self.__cfg['verification_label'])

    def train(self):
        r"""训练函数，每1000步保存模型，训练完毕使用验证集检验准确率。因为训练集过小，不用batch，每次计算都使用整个训练集
            
            训练次数：6000

            隐藏层固化率 0.25
        """
        # 启动会话
        with tf.Session() as sess:
            # 实例化计算图
            net = Network(datasize = self.train_data.shape[1],labelsize = self.train_label.shape[1])
            # 初始化变量
            sess.run(tf.global_variables_initializer())
            # 定义模型存储，最大储存10个节点
            saver = tf.train.Saver(max_to_keep=10)
            STEPS = 6000
            now_step = 0
            save_interval = 1000
            # 开始训练
            for i in range(STEPS):
                _, lossv = sess.run([net.train, net.loss],feed_dict={net.x: self.train_data, net.y_: self.train_label,net.keep_prob:0.75})
                # 取得当前训练步数
                now_step = sess.run(net.global_step)
                if (i + 1) % 10 == 0:
                    print('第{}步，当前loss：{:.4f}'.format(i + 1, lossv))
                
                if now_step % save_interval == 0:
                    saver.save(sess, self.CKPT_DIR + 'model', global_step=now_step)
            # 训练结束后验证结果
            accuracyv = sess.run(net.accuracy,feed_dict={net.x: self.verification_data, net.y_: self.verification_label, net.keep_prob:1})
            print("准确率:{:.4f}，共测试了{}张图片 ".format(accuracyv, len(self.verification_label)))
            print('finish')





