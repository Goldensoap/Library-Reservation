# _*_ coding:utf-8 _*_  
import tensorflow as tf 

class Network:
    r"""网络结构，2个隐藏层，1个输出层，第一个隐藏层有500个神经元，第二层有100个，输出层10个，并附带检验方法

    学习率0.01

    激活函数：Relu

    损失函数：交叉熵

    优化器：AdagradOptimizer（改良型梯度下降）
    """
    def __init__(self,datasize:int,labelsize:int)->None:
        r"""构造网络计算图,输入参数为训练集数据大小和对应标签（独热码）大小
        """
        self.data_size = datasize
        self.label_size = labelsize
        # 定义全局训练步数
        self.global_step = tf.Variable(0, trainable=False)
        # 定义学习率
        self.learning_rate = 0.01
        # 使用占位符定义标签和训练集张量节点
        self.x  = tf.placeholder(tf.float32, [None,self.data_size])
        self.y_ = tf.placeholder(tf.float32, [None, self.label_size])
        # 占位符定义固化比例（在一次计算中按比例固化参数，一般用在全连接网络中，目的是一定程度上防止过拟合，固化比例为1-keep）
        self.keep_prob = tf.placeholder(tf.float32)
        # 定义各层参数张量节点，隐藏层初始随机数
        w1 = tf.Variable(tf.truncated_normal([self.data_size,500],stddev = 0.1))
        w2 = tf.Variable(tf.truncated_normal([500,100],stddev = 0.1))
        w = tf.Variable(tf.zeros([100,self.label_size]))
        # 定义各层偏置张量节点，偏置初始为0
        b1 = tf.Variable(tf.zeros([500]))
        b2 = tf.Variable(tf.zeros([100]))
        b = tf.Variable(tf.zeros([self.label_size]))
        # 定义计算节点，矩阵运算，激活函数为relu
        a1 = tf.nn.relu(tf.matmul(self.x,w1)+b1)
        a1_drop = tf.nn.dropout(a1,self.keep_prob)
        a2 = tf.nn.relu(tf.matmul(a1_drop,w2)+b2)
        a2_drop = tf.nn.dropout(a2,self.keep_prob)
        # 定义输出层计算节点，使用softmax将结果映射到0-1之间的概率区间
        self.y = tf.nn.softmax(tf.matmul(a2_drop, w) + b)
        # 损失函数，使用交叉熵来表示运算结果与标签之间的差距
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_*tf.log(self.y),reduction_indices = [1]))
        # 优化器使用改良型梯度下降方法，此方法可以自动调整学习率等参数
        self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)
        # 定义预测方法
        self.predict = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.predict, "float"))

