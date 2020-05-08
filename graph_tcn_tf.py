
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,initializers

import numpy as np

class GraphNodes(keras.Model):
    def __init__(self,fout):
        super(GraphNodes,self).__init__()

        self.fout = fout # 16
        self.thresold = 1e-12 # eps

        self.fc = layers.Dense(self.fout,use_bias=False)
        self.leakyrelu = layers.LeakyReLU(alpha=0.2)
        self.initializer = initializers.GlorotUniform()
        self.bias = tf.Variable(self.initializer(shape=[self.fout], dtype=tf.float32))
        
    def euclidean_dist(self,x): # 求解欧式距离
        bs = x.shape[0] # bs
        x_m = tf.tile(tf.expand_dims(x,1),[1,bs,1]) # [bs,bs,2] 减数
        x_l = tf.tile(tf.expand_dims(x,0),[bs,1,1]) # [bs,bs,2] 被减数
        dist = tf.reduce_sum(tf.pow(x_l-x_m,2),2)
        dist = tf.sqrt(tf.clip_by_value(dist,self.thresold,tf.float32.max)) # 欧式距离
        dist_ = tf.exp(-1*dist) # 距离的负数为幂，e为底
        return dist_

    def get_adj(self,dist): # 求解领接矩阵,注意tensorflow中的tensor不能直接赋值
        bs = dist.shape[0] # bs
        A = tf.nn.softmax(dist,1).numpy() # 列归一化
        Alsum = A.sum(0) # 行求和
        Afinall = np.zeros((bs,bs),dtype=float) # 预先初始化最后的领接矩阵
        for i in range(bs):
            for j in range(bs):
                for k in range(bs):
                    Afinall[i][j] = Afinall[i][j] + A[i][k]*A[j][k]/Alsum[k]
        return tf.convert_to_tensor(Afinall,dtype=tf.float32) # 返回第一个领接矩阵

    def call(self,x): # x:[bs,2] 表示bs个行人，2表示(x,y) (相对位置)
        dist = self.euclidean_dist(x) # 欧式距离矩阵
        A = self.get_adj(dist) # 归一化的领接矩阵
        s = self.fc(x) # [bs,fout]
        h = self.leakyrelu(tf.matmul(A,s)+self.bias) # [bs,fout]
        return h

class MultiHeadGraphAttention(keras.Model): # 多头图注意力模型
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head # 头大小
        self.f_in = f_in # 输入大小
        self.f_out = f_out # 输出大小
        self.isbias = bias # 偏置
        self.initializer = initializers.GlorotUniform() # 初始化分布
        self.w = tf.Variable(self.initializer(shape=[self.n_head, self.f_in, self.f_out], dtype=tf.float32)) # 自定义参数 权重
        self.a_src = tf.Variable(self.initializer(shape=[self.n_head, self.f_out, 1], dtype=tf.float32)) # 自定义参数
        self.a_dst = tf.Variable(self.initializer(shape=[self.n_head, self.f_out, 1], dtype=tf.float32)) # 自定义参数

        self.leaky_relu = layers.LeakyReLU(alpha=0.2) # 激活函数
        self.softmax = layers.Softmax(axis=-1) # 归一层
        self.dropout = layers.Dropout(rate=attn_dropout) # Dropout 层
        if self.isbias:
            self.bias = tf.Variable(tf.zeros(self.f_out)) # 自定义参数 偏置

    def call(self, h): 
        bs = h.shape[0] # [bs]
        h_prime = tf.matmul(h, self.w) # [head,bs,f_out]
        attn_src = tf.matmul(h_prime, self.a_src) # [head,bs,1]
        attn_dst = tf.matmul(h_prime, self.a_dst) # [head,bs,1]
        attn = tf.tile(attn_src,[1,1,bs]) + tf.transpose(tf.tile(attn_dst,[1,1,bs]),[0, 2, 1]) # 每个行人对当前行人的影响
        # [head,bs,bs]
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = tf.matmul(attn, h_prime) # [head,bs,f_out]
        if self.isbias:
            output = output + self.bias
        output = tf.transpose(output,[1,0,2]) # [bs,head,f_out]
        output = tf.reshape(output,[bs,-1]) # [bs,head*f_out]
        return output

class ATTENTIONBLOCK(keras.Model):
	def __init__(self, n_units, n_heads, dropout=0.2):
		super(ATTENTIONBLOCK, self).__init__()
		self.n_layer = len(n_units) - 1 # 中间层大小
		self.dropout = dropout
		self.layer_stack = [] # 模型列表

		for i in range(self.n_layer):
			f_in = n_units[i] * n_heads[i - 1] if i else n_units[i] # 多头输入
			self.layer_stack.append( # 添加图注意力层
				MultiHeadGraphAttention(
					n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=self.dropout
				)
			)

	def call(self, x):
		for _, att_layer in enumerate(self.layer_stack):
			x = att_layer(x)
		return x

class EGAT(keras.Model):
    def __init__(self,n_units,n_heads):
        super(EGAT,self).__init__()
        self.n_units = n_units # [16,16,16]
        self.n_heads = n_heads # [4,1]
        
        self.att_block = ATTENTIONBLOCK(self.n_units,self.n_heads) # 注意力模型
        self.nodes = GraphNodes(self.n_units[0]) # 获得结点特征
        
    def call(self,x):
        x = self.nodes(x)
        x = self.att_block(x)
        return x

class TCN(keras.Model):
    def __init__(self,fout,n_layers=3,ksize=3):
        super(TCN,self).__init__()
        self.fout = fout # [48]
        self.n_layers = n_layers # [3]
        self.ksize = ksize # [3]
        self.paddings = tf.constant([[0,0],[self.ksize-1,0], [0, 0]])

        self.convs = [] # 空洞卷积
        for i in range(self.n_layers):
            self.convs.append(layers.Conv1D(self.fout,kernel_size=self.ksize))

    def call(self,x):
        for _, conv_layer in enumerate(self.convs):
            # left padding
            x = tf.pad(x, self.paddings, "CONSTANT")
            x = conv_layer(x)
        return x

class Encoder(keras.Model): # 这里使用残差连接
    def __init__(self,fout,n_layers=3,ksize=3):
        super(Encoder,self).__init__()
        self.fout = fout # [48]
        self.n_layers = n_layers # [3]
        self.ksize = ksize # [3]

        self.tcnf = TCN(self.fout,self.n_layers,self.ksize)
        self.tcng = TCN(self.fout,self.n_layers,self.ksize)

    def call(self,x):
        residual = x
        f = tf.sigmoid(self.tcnf(x))
        g = tf.tanh(self.tcng(x))
        return residual+f*g # residual connection

class EncoderBlock(keras.Model): # 这里使用跳连接
    def __init__(self,fout,blocks=3,n_layers=3,ksize=3):
        super(EncoderBlock,self).__init__()
        self.fout = fout # [48]
        self.blocks = blocks # [3]
        self.n_layers = n_layers # [3]
        self.ksize = ksize # [3]

        self.encoders = [] # 空洞卷积
        for i in range(self.blocks):
            self.encoders.append(Encoder(self.fout,self.n_layers,self.ksize)) # 添加多个Encoder模块

    def call(self,x):
        s = 0
        for _, en_layer in enumerate(self.encoders):
            x = en_layer(x)
            s = s + x # skip connection
        return s

class Embedding(keras.Model): # 这里使用绝对位置，对位置进行编码
    def __init__(self,fout):
        super(Embedding,self).__init__()
        self.fout = fout # [32]
        self.fc = layers.Dense(self.fout,use_bias=False)

    def call(self,x):
        return self.fc(x)

class Decoder(keras.Model): # 生成相对位置
    def __init__(self,pre_len,fout,noise_dim,M=20):
        super(Decoder,self).__init__()
        self.pre_len = pre_len # 8
        self.fout = fout # 2
        self.noise_dim = noise_dim # 16
        self.M = M # 20

        self.mlp = keras.Sequential() # 多层感知机
        self.mlp.add(layers.Dense(390))
        self.mlp.add(layers.LeakyReLU(alpha=0.2))
        self.mlp.add(layers.Dense(204))
        self.mlp.add(layers.LeakyReLU(alpha=0.2))
        self.mlp.add(layers.Dense(self.pre_len*self.fout))
        self.mlp.add(layers.LeakyReLU(alpha=0.2))

    def call(self,x):
        Y = []
        for i in range(self.M): # 预测多条合理的轨迹
            noise = tf.random.normal([x.shape[0],self.noise_dim]) # 随机噪声
            h = tf.concat([x,noise],1) # 添加噪声
            h = tf.reshape(self.mlp(h),[-1,self.pre_len,self.fout]) # [bs,pre,2]
            Y.append(h)
        return tf.stack(Y,0) # [M,bs,pre,2]

class GraphTCN(keras.Model): # 这里使用跳连接
    def __init__(self,fin,fout,n_units,n_heads,pre_len,noise_dim,M,blocks,n_layers,ksize):
        super(GraphTCN,self).__init__()
        self.fin = fin # 2 x,y
        self.fout = fout # [32] 绝对位置嵌入维度
        self.n_units = n_units # [16,16,16]
        self.n_heads = n_heads # [4,1]
        self.pre_len = pre_len # 8
        self.noise_dim = noise_dim #16
        self.M = M # 20
        self.blocks = blocks # [3]
        self.n_layers = n_layers # [3]
        self.ksize = ksize # [3]

        self.egat = EGAT(self.n_units,self.n_heads) # 相对位置嵌入
        self.embedding = Embedding(self.fout) # 绝对位置嵌入
        self.encoder = EncoderBlock(self.fout+self.n_units[-1],self.blocks,self.n_layers,self.ksize) # 编码器
        self.decoder = Decoder(self.pre_len,self.fin,self.noise_dim,self.M) # 解码器

    def call(self,xre,xab): # xre,xab [len,bs,2]
        bs = xre.shape[1]
        egats, embeddings = [],[]
        for i in range(xre.shape[0]):
            egats.append(self.egat(xre[i]))
            embeddings.append(self.embedding(xab[i]))
        egats = tf.stack(egats,-1) # [bs,16,len]
        embeddings = tf.stack(embeddings,-1) # [bs,32,len]
        x = tf.concat([egats,embeddings],1) # [bs,48,len]
        x = self.encoder(tf.transpose(x,[0,2,1])) # [bs,hs,len]
        x = self.decoder(tf.reshape(x,[bs,-1])) # [M,bs,pre,2]
        return x


g = GraphTCN(fin=2,fout=32,n_units=[16,16,16], n_heads=[4,1],pre_len=8,
            noise_dim=16,M=20,blocks=3,n_layers=3,ksize=3)

xre = tf.random.normal([12,5,2])
xab = tf.random.normal([12,5,2])
print(g(xre,xab).shape)
