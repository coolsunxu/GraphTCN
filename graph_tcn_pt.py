
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNodes(nn.Module):
    def __init__(self,fin,fout):
        super(GraphNodes,self).__init__()

        self.fin = fin # 2
        self.fout = fout # 16
        self.thresold = 1e-12 # eps

        self.fc = nn.Linear(self.fin,self.fout,bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.bias = nn.Parameter(torch.Tensor(self.fout)) # 自定义参数 偏置
        
    def euclidean_dist(self,x): # 求解欧式距离
        bs = x.shape[0] # bs
        x_m = x.unsqueeze(1).repeat(1,bs,1) # [bs,bs,2] 减数
        x_l = x.unsqueeze(0).repeat(bs,1,1) # [bs,bs,2] 被减数
        dist = torch.pow(x_l-x_m,2).sum(2).clamp(min=self.thresold).sqrt() # 欧式距离
        dist_ = torch.exp(-1*dist) # 距离的负数为幂，e为底
        return dist_

    def get_adj(self,dist): # 求解领接矩阵
        bs = dist.shape[0] # bs
        A = F.softmax(dist,1) # 列归一化
        Alsum = A.sum(0) # 行求和
        Afinall = torch.zeros(bs,bs,dtype=torch.float32) # 预先初始化最后的领接矩阵
        for i in range(bs):
            for j in range(bs):
                for k in range(bs):
                    Afinall[i][j] = Afinall[i][j] + A[i][k]*A[j][k]/Alsum[k]
        return Afinall # 返回第一个领接矩阵

    def forward(self,x): # x:[bs,2] 表示bs个行人，2表示(x,y) (相对位置)
        dist = self.euclidean_dist(x) # 欧式距离矩阵
        A = self.get_adj(dist) # 归一化的领接矩阵
        s = self.fc(x) # [bs,fout]
        h = self.leakyrelu(torch.matmul(A,s)+self.bias) # [bs,fout]
        return h

class MultiHeadGraphAttention(nn.Module): # 多头图注意力模型
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head # 头大小
        self.f_in = f_in # 输入大小
        self.f_out = f_out # 输出大小
        self.isbias = bias # 偏置
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out)) # 自定义参数 权重
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1)) # 自定义参数
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1)) # 自定义参数

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2) # 激活函数
        self.softmax = nn.Softmax(dim=-1) # 归一层
        self.dropout = nn.Dropout(attn_dropout) # Dropout 层
        self.bias = nn.Parameter(torch.Tensor(f_out)) # 自定义参数 偏置
        nn.init.constant_(self.bias, 0) # 初始化参数

        # 初始化自定义参数
        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h): 
        bs = h.shape[0] # [bs]
        h_prime = torch.matmul(h, self.w) # [head,bs,f_out]
        attn_src = torch.matmul(h_prime, self.a_src) # [head,bs,1]
        attn_dst = torch.matmul(h_prime, self.a_dst) # [head,bs,1]
        attn = attn_src.expand(-1, -1, bs) + attn_dst.expand(-1, -1, bs).permute( # 每个行人对当前行人的影响
            0, 2, 1
        ) # [head,bs,bs]
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime) # [head,bs,f_out]
        if self.isbias:
            output = output + self.bias
        output = output.permute(1,0,2) # [bs,head,f_out]
        output = torch.reshape(output,(bs,-1)) # [bs,head*f_out]
        return output

class ATTENTIONBLOCK(nn.Module):
	def __init__(self, n_units, n_heads, dropout=0.2):
		super(ATTENTIONBLOCK, self).__init__()
		self.n_layer = len(n_units) - 1 # 中间层大小
		self.dropout = dropout
		self.layer_stack = nn.ModuleList() # 模型列表

		for i in range(self.n_layer):
			f_in = n_units[i] * n_heads[i - 1] if i else n_units[i] # 多头输入
			self.layer_stack.append( # 添加图注意力层
				MultiHeadGraphAttention(
					n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
				)
			)

	def forward(self, x):
		for _, att_layer in enumerate(self.layer_stack):
			x = att_layer(x)
		return x

class EGAT(nn.Module):
    def __init__(self,fin,n_units,n_heads):
        super(EGAT,self).__init__()
        self.fin = fin # [2]
        self.n_units = n_units # [16,16,16]
        self.n_heads = n_heads # [4,1]
        
        self.att_block = ATTENTIONBLOCK(self.n_units,self.n_heads) # 注意力模型
        self.nodes = GraphNodes(self.fin,self.n_units[0]) # 获得结点特征
        
    def forward(self,x):
        x = self.nodes(x)
        x = self.att_block(x)
        return x

class TCN(nn.Module):
    def __init__(self,fin,fout,layers=3,ksize=3):
        super(TCN,self).__init__()
        self.fin = fin # [48]
        self.fout = fout # [48]
        self.layers = layers # [3]
        self.ksize = ksize # [3]

        self.convs = nn.ModuleList() # 空洞卷积
        for i in range(self.layers):
            self.convs.append(nn.Conv1d(self.fin,self.fout,kernel_size=self.ksize))

    def forward(self,x):
        for _, conv_layer in enumerate(self.convs):
            # left padding
            x = nn.functional.pad(x,(self.ksize-1,0))
            x = conv_layer(x)
        return x

class Encoder(nn.Module): # 这里使用残差连接
    def __init__(self,fin,fout,layers=3,ksize=3):
        super(Encoder,self).__init__()
        self.fin = fin # [48]
        self.fout = fout # [48]
        self.layers = layers # [3]
        self.ksize = ksize # [3]

        self.tcnf = TCN(self.fin,self.fout,self.layers,self.ksize)
        self.tcng = TCN(self.fin,self.fout,self.layers,self.ksize)

    def forward(self,x):
        residual = x
        f = torch.sigmoid(self.tcnf(x))
        g = torch.tanh(self.tcng(x))
        return residual+f*g # residual connection

class EncoderBlock(nn.Module): # 这里使用跳连接
    def __init__(self,fin,fout,blocks=3,layers=3,ksize=3):
        super(EncoderBlock,self).__init__()
        self.fin = fin # [48]
        self.fout = fout # [48]
        self.blocks = blocks # [3]
        self.layers = layers # [3]
        self.ksize = ksize # [3]

        self.encoders = nn.ModuleList() # 空洞卷积
        for i in range(self.blocks):
            self.encoders.append(Encoder(self.fin,self.fout,self.layers,self.ksize)) # 添加多个Encoder模块

    def forward(self,x):
        s = 0
        for _, en_layer in enumerate(self.encoders):
            x = en_layer(x)
            s = s + x # skip connection
        return s

class Embedding(nn.Module): # 这里使用绝对位置，对位置进行编码
    def __init__(self,fin,fout):
        super(Embedding,self).__init__()
        self.fin = fin # [2]
        self.fout = fout # [32]

        self.fc = nn.Linear(self.fin,self.fout,bias=False)

    def forward(self,x):
        return self.fc(x)

class Decoder(nn.Module): # 生成相对位置
    def __init__(self,obs_len,pre_len,fin,fout,noise_dim,M=20):
        super(Decoder,self).__init__()
        self.obs_len = obs_len # 12
        self.pre_len = pre_len # 8
        self.fin = fin # 48
        self.fout = fout # 2
        self.noise_dim = noise_dim # 16
        self.M = M # 20

        self.mlp = nn.Sequential( # 多层感知机
                nn.Linear(self.fin*self.obs_len+self.noise_dim,390),
                nn.LeakyReLU(0.2),
                nn.Linear(390,204),
                nn.LeakyReLU(0.2),
                nn.Linear(204,self.pre_len*self.fout),
                nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        Y = []
        for i in range(self.M): # 预测多条合理的轨迹
            noise = torch.randn(x.shape[0],self.noise_dim) # 随机噪声
            h = torch.cat((x,noise),1) # 添加噪声
            h = torch.reshape(self.mlp(h),(-1,self.pre_len,self.fout)) # [bs,pre,2]
            Y.append(h)
        return torch.stack(Y,0) # [M,bs,pre,2]


class GraphTCN(nn.Module): # 这里使用跳连接
    def __init__(self,fin,fout,n_units,n_heads,obs_len,pre_len,noise_dim,M,blocks,n_layers,ksize):
        super(GraphTCN,self).__init__()
        self.fin = fin # 2 x,y
        self.fout = fout # [32] 绝对位置嵌入维度
        self.n_units = n_units # [16,16,16]
        self.n_heads = n_heads # [4,1]
        self.obs_len = obs_len # 12
        self.pre_len = pre_len # 8
        self.noise_dim = noise_dim #16
        self.M = M # 20
        self.blocks = blocks # [3]
        self.n_layers = n_layers # [3]
        self.ksize = ksize # [3]

        self.egat = EGAT(self.fin,self.n_units,self.n_heads) # 相对位置嵌入
        self.embedding = Embedding(self.fin,self.fout) # 绝对位置嵌入
        self.encoder = EncoderBlock(self.fout+self.n_units[-1],self.fout+self.n_units[-1],self.blocks,self.n_layers,self.ksize) # 编码器
        self.decoder = Decoder(self.obs_len,self.pre_len,self.fout+self.n_units[-1],self.fin,self.noise_dim,self.M) # 解码器

    def forward(self,xre,xab): # xre,xab [len,bs,2]
        bs = xre.shape[1]
        egats, embeddings = [],[]
        for i in range(xre.shape[0]):
            egats.append(self.egat(xre[i]))
            embeddings.append(self.embedding(xab[i]))
        egats = torch.stack(egats,-1) # [bs,16,len]
        embeddings = torch.stack(embeddings,-1) # [bs,32,len]
        x = torch.cat((egats,embeddings),1) # [bs,48,len]
        x = self.encoder(x) # [bs,hs,len]
        x = self.decoder(torch.reshape(x,(bs,-1))) # [M,bs,pre,2]
        return x


g = GraphTCN(fin=2,fout=32,n_units=[16,16,16], n_heads=[4,1],obs_len=12,pre_len=8,
            noise_dim=16,M=20,blocks=3,n_layers=3,ksize=3)

xre = torch.randn(12,5,2)
xab = torch.randn(12,5,2)
print(g(xre,xab).shape)
