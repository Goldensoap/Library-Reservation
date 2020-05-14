# _*_ coding:utf-8 _*_  
import numpy as np
import tensorflow as tf
from .model import Network
from ..image_process import Image_process
from ..custom_tools import Config_Tools

class Predict:
    r"""预测类，提供预测函数，载入持久化模型，提供预测结果
    """
    def __init__(self):
        r"""实例化计算图和图片处理，载入配置
        """
        self.__cfgtool = Config_Tools()
        self.__cfg = self.__cfgtool.cfg_to_dict('NN')
        self.net = Network(datasize = 4900,labelsize = 10)
        self.CKPT_DIR = self.__cfg['cpkt_dir']
        self.image_process = Image_process()

    def predict(self, image_path:str)->str:
        r"""预测函数，image_path是图片的路径，以字符串形式返回预测结果
        """
        result = ''
        # 打开会话
        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())
            #定义储存器以及获取检查点
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.CKPT_DIR)
            # 如果检查点存在，载入持久化模型
            if ckpt and ckpt.model_checkpoint_path:
                path = self.CKPT_DIR+ckpt.model_checkpoint_path.split('/')[-1]
                saver.restore(sess, path)
            else:
                raise FileNotFoundError("未保存任何模型")
            # 图像处理成合适输入网络的形式（一个验证码包含4个数字，处理+切割，获得有序的4个数字图片）
            xlist = self.image_process.get_NN_data(image_path)
            # 依次预测
            for i in xlist:
                x = [i]
                y = sess.run(self.net.y,feed_dict = {self.net.x:x,self.net.keep_prob:1})
                result += str(np.argmax(y[0]))
        return result
        
