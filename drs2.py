from cgi import print_arguments, print_form
from cmath import tanh
from re import I
# from math import dist
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
# from One_hot_encoder import One_hot_encoder
import numpy as np
import gatnetSTTNs as NEW_GAT
from torch_geometric.nn import JumpingKnowledge


# model input shape:[1, N, T]
# model output shape:[N, T]
class STGNSNet(nn.Module):
    # 构造函数，实例化时自动调用
    def __init__(self, device, adj, corr, dist, in_channels, batch_size, embed_size, time_num,
                 num_layers, gat_layers, mode, T_dim, output_T_dim, heads, dropout, forward_expansion):

        self.mode = mode
        self.num_layers = num_layers
        self.gat_layers = gat_layers
        super(STGNSNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.transformer = Transformer(device, batch_size, embed_size, heads, adj, corr,
                                       dist, num_layers, gat_layers, mode, time_num, dropout, forward_expansion, T_dim)
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv3 = nn.Conv2d(embed_size, in_channels, 1)
    # 可被隐式调用，原因是python的__call__函数

    def forward(self, x, device):  # x是输入的data batch 还没有输入device
        # input x:[B,N,T,C]
        # x = x.unsqueeze(0)  # [1, C, N, T]---------------------------------
        data = x["flow_x"].to(device)  # data batch  将data部署到device[64,5,12,1]
        # 由于要使用GAT图注意力网络，因此要储存图信息
        graph = x["graph"][0].to(device)
        corr = x["corr"][0].to(device)
        dist = x["dist"][0].to(device)
        # [B,N,T,C]-->[B C N T] ------------------------
        data_conv = data.permute(0, 3, 1, 2)
        # 通道变换
        # [B C N T] -->[B, embed_size, N, T][64,64,5,12]
        data_conv = self.conv1(data_conv)
        # print("x.shape=",data_conv.shape)
        # ---x = x.squeeze(0)  # [1,embed_size, N, T]-------------------------------------------
        # [B, embed_size, N, T]-->[B,N, T, embed_size][64, 5, 12, 64]
        data_trans = data_conv.permute(0, 2, 3, 1)
        # 调用Transformer的forward
        data_trans = self.transformer(
            data_trans, graph, corr, dist, device)  # [B,N, T, embed_size]
        # print("data_trans shape:", data_trans.shape)
        # 预测时间T_dim，转换时间维数
        # ---x = x.unsqueeze(0)  # [1, N, T, C], C = embed_size----------------------------------------------
        data_cov2 = data_trans.permute(0, 2, 1, 3)  # [B, T, N, C]
        data_cov2 = self.conv2(data_cov2)  # [B, out_T_dim, N, C]
        # print("STTN conv2 shape", data_cov2.shape)
        # 将通道降为in_channels
        data_cov3 = data_cov2.permute(0, 3, 2, 1)  # [B, C, N, out_T_dim]
        data_cov3 = self.conv3(data_cov3)  # [B, in_channels, N, out_T_dim]
        data_final = data_cov3.permute(0, 2, 3, 1)
        # out = x.unsqueeze(0).unsqueeze(0)
        # print("STTN finnal shape", data_final.shape)
        return data_final


class Transformer(nn.Module):
    def __init__(self, device, batch_size, embed_size, heads, adj, corr, dist, num_layers, gat_layers, mode, time_num, dropout, forward_expansion, T_dim):
        super(Transformer, self).__init__()
        self.num_layers = num_layers

        self.sttnblock = STGNSNetBlock(device, batch_size, embed_size, heads, adj, corr,
                                       dist, gat_layers, mode, time_num, dropout, forward_expansion, T_dim)

    def forward(self, data_trans, graph, corr, dist, device):
        # 给Wq、Wk、Wv赋值
        q, k, v = data_trans, data_trans, data_trans
        for i in range(self.num_layers):
            # 调用STGNSNetBlock的forward
            out = self.sttnblock(q, k, v, graph, corr, dist, device)
            q, k, v = out, out, out

        return out


# model input:[N, T, C]
# model output[N, T, C]
class STGNSNetBlock(nn.Module):
    def __init__(self, device, batch_size, embed_size, heads, adj, corr, dist, gat_layers, mode, time_num, dropout, forward_expansion, T_dim):
        super(STGNSNetBlock, self).__init__()
        self.SpatialTansformer = STransformer(
            device, embed_size, heads, adj, corr, dist, gat_layers, mode, dropout, forward_expansion)
        self.TemporalTransformer = TTransformer(
            embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.route_laters = gat_layers+1
        self.liner_first = nn.Linear(embed_size, embed_size)
        self.liner_second = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_size = batch_size
        self.softmax = nn.Softmax(dim=-1)
        # self.embed_size = embed_size
        self.useGate = 1  # 控制是否使用动态路由门控
        self.if_fusion = torch.nn.Parameter(
            torch.randint(0, 2, (3, gat_layers+1)).float())
        # self.if_fusion = torch.nn.Parameter(
        #     torch.ones(0, 2, (3, gat_layers+1)).float())
        # self.threshold_value = torch.nn.Parameter(torch.zeros(1))

        self.gate_conv_beta = nn.Sequential(
            nn.Conv2d(len(adj), 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 3, 1, 1, 0, bias=True),
        )

        self.gate_ac = nn.Sequential(
            nn.Tanh(),
            nn.ReLU(),
        )
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(5, 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

    def forward(self, query, key, value, graph, corr, dist, device):
        # 把query的四个元组分开，赋给B、N、T、C (64 5 12 64)
        B = query.shape[0]  # ----------------------------------
        N = query.shape[1]  # ----------------------------------
        T = query.shape[2]  # ----------------------------------
        C = query.shape[3]
        # 保留jkgat中每一步的结果 out_g:连通性图  out_c:相关性图  out_d:距离图
        out_g = self.SpatialTansformer(
            query, key, value, graph, "graph", device)  # (64 5 12 64)
        out_c = self.SpatialTansformer(query, key, value, corr, "corr", device)
        out_d = self.SpatialTansformer(query, key, value, dist, "dist", device)
        # 从连通性图出发开始路由选择
        # out1 = out_g[0]+out_c[0]+out_d[0]
        # 从第二层，即i=1开始进行路由选择

        for i in range(self.route_laters):
            # print(i)
            if(i == 0):
                out1 = out_g[0]+out_c[0]+out_d[0]
            # print(self.useGate)
            if self.useGate:
                # print("使用门控")
                ww = self.gate_conv_beta(out1).view(
                    self.batch_size, 1, 3).permute(2, 0, 1).contiguous()
                ww = self.gate_ac(ww)
            else:
                # print("不使用门控")
                ww = torch.ones(3, self.batch_size, 1).to(device)

            feat_p = torch.mul(ww[0], self.softmax(self.if_fusion[0][i] * out_g[i]).view(
                self.batch_size, -1)).view(self.batch_size, -1, out_g[i].shape[2], out_g[i].shape[3])+out_g[i]
            feat_c = torch.mul(ww[1], self.softmax(self.if_fusion[1][i]*out_c[i]).view(
                self.batch_size, -1)).view(self.batch_size, -1, out_c[i].shape[2], out_c[i].shape[3])+out_c[i]
            feat_d = torch.mul(ww[2], self.softmax(self.if_fusion[2][i]*out_d[i]).view(
                self.batch_size, -1)).view(self.batch_size, -1, out_d[i].shape[2], out_d[i].shape[3])+out_d[i]
            out1 = out1+feat_p+feat_c+feat_d
        out1 = torch.sigmoid(out1)
        out1 = self.dropout(self.norm1(out1))
        out1 = self.liner_first(self.norm1(out1)+query)
        # print('使用单图')
        # out1 = self.liner_first(self.norm1(out_g)+query)
        # out1 = self.liner_first(self.norm1(out_c)+query)
        # out1 = self.liner_first(self.norm1(out_d)+query)
        out2 = self.liner_second(self.norm2(
            self.TemporalTransformer(query, key, value, device)+query))

        # ==========经历过时空模块的门控==========
        # 得到两个模块的权重
        GateS = torch.sigmoid(out1)
        GateT = torch.sigmoid(out2)
        out3 = GateS * out1 + GateT * out2
        return out3

# model input:[N, T, C]
# model output:[N, T, C]


class STransformer(nn.Module):
    def __init__(self, device, embed_size, heads, adj, corr, dist, gat_layers, mode, dropout, forward_expansion):
        super(STransformer, self).__init__()
        self.num_layers = gat_layers
        self.adj = adj
        self.corr = corr
        self.dist = dist
        self.D_S = adj
        # 类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.D_S = nn.Parameter(adj)
        self.embed_linear = nn.Linear(adj.shape[0], embed_size)
        self.attention = SSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        '''
        #==========将GCN换做GAT来处理有向图#==========
        #sttns结构：embed_linear、attention、norm1、feed_forward、norm2+GCN
        #改为embed_linear、attention、norm1、feed_forward、norm2+GAT
         '''
        self.gat = NEW_GAT.GATNet(embed_size, embed_size * 2, embed_size, 2)
        self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

        self.if_add = torch.nn.Parameter(torch.ones(gat_layers))
        '''
        # ==========将GCN换做GAT来处理有向图#==========
        # sttns结构：embed_linear、attention、norm1、feed_forward、norm2+GCN
        # 改为embed_linear、attention、norm1、feed_forward、norm2+GAT
         '''
        for i in range(1, self.num_layers):
            setattr(self, 'gat{}'.format(i), NEW_GAT.GATNet(
                embed_size, embed_size * 2, embed_size, 2))
        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(embed_size, embed_size)
        elif mode == 'cat':
            self.fc = nn.Linear(self.num_layers * embed_size, embed_size)

    def dr_jkgat(self, query, graph, type, device):
        if(type == "graph"):
            adj_matrix = self.adj
        elif (type == "corr"):
            adj_matrix = self.corr
        else:
            adj_matrix = self.dist
        # jk-GAT 部分
        layer_out = []  # 保存每一层的结果[128, 5, 12, 64]
        for i in range(self.num_layers):
            # size=(64, 5, 0, 64)
            Y_G = torch.Tensor(
                query.shape[0], query.shape[1], 0, query.shape[3])
            for t in range(query.shape[2]):  # 分12次计算
                o = self.gat(query[:, :, t, :], graph,
                             adj_matrix.to(device))
                Y_G = Y_G.to(device)
                Y_G = torch.cat((o, Y_G), dim=2)  # 不同的是第三维，即dim=2

            # X_G = self.dropout(self.norm1(Y_G))

            query = torch.sigmoid(Y_G)
            # query =torch.relu(Y_G)
            layer_out.append(query)
            layer_out.append(Y_G)

        h = self.jk(layer_out)  # JK层
        h = self.fc(h)
        out = h
        out = torch.sigmoid(h)
        out = self.dropout(self.norm1(out))
        # 1.路由空间使用
        # layer_out.append(out)
        return layer_out  # layer_out是一个list

        # 2.不使用路由空间
        # print("单图")
        # return out #out是一个tensor

    def forward(self, query, key, value, graph, type, device):
        # Spatial Embedding 部分  q k v 的维度 [B,N, T, embed_size]
        # DS为空间位置编码 空间位置编码需要batch吗？
        # 把query的四个元组分开，赋给B、N、T、C(64 5 12 64)
        B = query.shape[0]  # ----------------------------------64
        N = query.shape[1]  # ----------------------------------5
        T = query.shape[2]  # ----------------------------------12
        C = query.shape[3]  # ----------------------------------64
        # print("query.shape",query.shape)
        D_S = self.embed_linear((self.D_S.to(device)))
        #讲DS扩展到(T, N, C)
        D_S = D_S.expand(T, N, C)
        D_S = D_S.permute(1, 0, 2)  # [N T C]
        # unsqueeze(dim)就是在维度序号为dim的地方给tensor增加一维
        D_S = D_S.unsqueeze(0)
        # print("DS shape",D_S.shape)  torch.Size([1, 5, 12, 64])

        X_G = self.dr_jkgat(query, graph, type, device)  # 在动态路由下，X_G是一个list
        # spatial transformer
        # 位置嵌入
        query = query + D_S
        # print("q+DS",query.shape)
        value = value + D_S
        key = key + D_S

        # 自注意力模块
        attn = self.attention(value, key, query)  # [N, T, C]
        M_s = self.dropout(self.norm1(attn + query))

        # 对应feedforward network
        feedforward = self.feed_forward(M_s)
        U_s = self.dropout(self.norm2(feedforward + M_s))

        # 融合，类似于门控，对应~
        # 1. 动态路由，此时out是一个list
        out = []
        for i in range(len(X_G)):
            g = torch.sigmoid(self.fs(U_s) + self.fg(X_G[i]))
            out.append(g * U_s + (1 - g) * X_G[i])
        # print("len(out):", len(out))
        return out

        # 2. 不使用动态路由，此时out是一个tensor
        # g = torch.sigmoid(self.fs(U_s) + self.fg(X_G))
        # out = g * U_s + (1 - g) * X_G
        # return out


# model input:[N,T,C]
# model output:[N,T,C]


class SSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.values = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.queries = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.keys = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        B, N, T, C = query.shape
        query = query.reshape(B, N, T, self.heads, self.per_dim)
        keys = keys.reshape(B, N, T, self.heads, self.per_dim)
        values = values.reshape(B, N, T, self.heads, self.per_dim)
        #print("Q shape",query.shape)
        # q, k, v:[N, T, heads, per_dim]
        queries = self.queries(query)
        keys = self.keys(keys)
        values = self.values(values)
        #print("Qs shape",queries.shape)
        # spatial self-attention
        attn = torch.einsum("bqthd, bkthd->bqkth",
                            (queries, keys))  # [N, N, T, heads]

        # Attention(Q,K,V)=(softmax(Q(KT))/sqrt(dk))V
        attention = torch.softmax(attn / (self.embed_size ** (1 / 2)), dim=1)
        #print("attenton shape",attention.shape)
        # [N, T, heads, per_dim]
        out = torch.einsum("bqkth,bkthd->bqthd", (attention, values))
        out = out.reshape(B, N, T, self.heads * self.per_dim)  # [N, T, C]
        #print("out shape",out.shape)
        out = self.fc(out)

        return out

# input[N, T, C]


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        # Temporal embedding One hot
        self.time_num = time_num
        # self.one_hot = One_hot_encoder(embed_size, time_num)
        # # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(
            time_num, embed_size)  # temporal embedding选用nn.Embedding

        self.attention = TSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, device):
        # q, k, v：[N, T, C]
        B, N, T, C = query.shape

        # D_T = self.one_hot(t, N, T)  # temporal embedding选用one-hot方式 或者
        # D_T是时间位置编码 需要batch吗？

        # 时间信息嵌入
        D_T = self.temporal_embedding(torch.arange(0, T).to(
            device))  # temporal embedding选用nn.Embedding

        D_T = D_T.expand(N, T, C)
        D_T = D_T.unsqueeze(0)
        # TTransformer
        x = D_T + query

        # 自注意力模块
        attention = self.attention(x, x, x)
        M_t = self.dropout(self.norm1(attention + x))
        # 反向传播
        feedforward = self.feed_forward(M_t)
        # 融合
        U_t = self.dropout(self.norm2(M_t + feedforward))
        out = U_t + x + M_t
        return out


class TSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = self.embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # q, k, v:[N, T, C]
        B, N, T, C = query.shape

        # q, k, v:[N,T,heads, per_dim]
        keys = key.reshape(B, N, T, self.heads, self.per_dim)
        queries = keys
        values = keys
        # queries = query.reshape(B,N, T, self.heads, self.per_dim)
        # values = value.reshape(B,N, T, self.heads, self.per_dim)

        keys = self.keys(keys)
        values = keys
        queries = keys
        # values = self.values(keys)
        # queries = self.queries(keys)

        # compute temperal self-attention
        attnscore = torch.einsum(
            "bnqhd, bnkhd->bnqkh", (queries, keys))  # [N, T, T, heads]
        attention = torch.softmax(
            attnscore / (self.embed_size ** (1/2)), dim=2)

        # [N, T, heads, per_dim]
        out = torch.einsum("bnqkh, bnkhd->bnqhd", (attention, values))
        out = out.reshape(B, N, T, self.embed_size)
        out = self.fc(out)

        return out
