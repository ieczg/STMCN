# @Time    : 2020/8/25
# @Author  : LeronQ
# @github  : https://github.com/LeronQ

import numpy as np
import csv
import torch
import math
from torch.utils import data
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np
from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft

#返回e^(-(x^2 /sigmoid^2))


def gaussuain(x, sigmoid):
    molecule = x * x
    demominator = sigmoid * sigmoid
    #left = 1 / (math.sqrt(2 * math.pi) * sigmoid)
    print(math.exp(-molecule/demominator))
    return math.exp(-molecule/demominator)

# 返回标准化的邻接矩阵。


def get_normalized_adj(A):
    """
    返回标准化的邻接矩阵。
    Returns the degree normalized adjacency matrix.
    """
    # np.diag()如果是2D数组，返回k位置的对角线。如果是1D数组，返回一个v作为k位置对角线的2维数组
    # np.ones()函数返回给定形状和数据类型的新数组，其中元素的值设置为1。
    # np.sum(a, axis=0) ------->列求和  np.sum(a, axis=1) ------->行求和
    # reshape(-1, m)即列数固定，行数需要计算
    # np.sqrt(D) 数组每个元素返回一个非负平方根
    # numpy.reciprocal() 函数返回参数逐元素的倒数
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(
        diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_wave


def get_corr_matrix(file):
    f = []
    alpha = 7000      # moderate bandwidth constraint
    tau = 0.1          # noise-tolerance (no strict fidelity enforcement)
    K = 3        # 3 modes
    DC = 0             # no DC part imposed
    init = 0           # initialize omegas uniformly
    tol = 1e-7
    low_fre = []  # 存储低频数据
    inter_fre = []  # 存储中频数据
    high_fre = []  # 存储高低频数据
    for i in range(len(file)):
        # print(i)
        f.append(pd.read_excel("oil_data/"+file[i]+".xlsx", header=None))
        u, u_hat, omega = VMD(f[i].values, alpha, tau, K, DC, init, tol)
        low_fre.append(u[0])
        inter_fre.append(u[1])
        high_fre.append(u[2])
    # 低频
    data1 = pd.DataFrame(low_fre)
    corr1 = data1.T.corr(method='pearson')
    # print(corr1)
    # 中频
    data2 = pd.DataFrame(inter_fre)
    corr2 = data2.T.corr(method='pearson')
    # print(corr2)
    # 高频
    data3 = pd.DataFrame(high_fre)
    corr3 = data3.T.corr(method='pearson')
    # print(corr3)
    # 低频与中频相加(去掉高频)
    corr1 += corr2

    # 低频与中频做哈达玛乘积
    corr1 = corr1.values
    for j in range(len(corr1)):
        for k in range(len(corr1[j])):
            if (corr1[j][k] < 0):
                corr1[j][k] = 0
    corr1 -= np.diag([1] * len(file))
    corr1 = get_normalized_adj(corr1)
    return corr1


def get_dist_matrix(distance_file: str, num_nodes: int) -> np.array:
    if(distance_file == "adjacent_matrix/adjacent_matrix.csv"):
        file = "adjacent_matrix/dist1.csv"
    A = np.zeros([int(num_nodes), int(num_nodes)])  # 构造全0的邻接矩阵
    with open(file, "r") as f_d:
        reader = csv.reader(f_d)  # 读取.csv文件.
        for item in reader:   # 将一行给item组成列表
            # i, j, distance = int(item[0]), int(item[1]), float(item[2])
            i = int(item[0])
            j = int(item[1])
            distance = float(item[2])

            A[i, j], A[j, i] = distance, distance
    A -= np.diag(np.ones(A.shape[0], dtype=np.float32))
    A = get_normalized_adj(A)
    # print("dist:",A.shape)
    return A

# 获取邻接矩阵


def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="distance") -> np.array:
    """
    :param distance_file: str, path of csv file to save the distances between nodes.
    :param num_nodes: int, number of nodes in the graph
    :param id_file: str, path of txt file to save the order of the nodes.就是排序节点的绝对编号所用到的，这里排好了，不需要
    :param graph_type: str, ["connect", "distance"]，这个就是考不考虑节点之间的距离
    :return:
        np.array(N, N)


    """
    A = np.zeros([int(num_nodes), int(num_nodes)])  # 构造全0的邻接矩阵

    if id_file:  # 就是给节点排序的绝对文件，这里是None，则表示不需要
        with open(id_file, "r") as f_id:
            # 将绝对编号用enumerate()函数打包成一个索引序列，然后用node_id这个绝对编号做key，用idx这个索引做value
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(
                f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline()  # 表头，跳过第一行.
                reader = csv.reader(f_d)  # 读取.csv文件.
                for item in reader:   # 将一行给item组成列表
                    if len(item) != 3:  # 长度应为3，不为3则数据有问题，跳过（from to dict）
                        continue
                    i, j, distance = int(item[0]), int(
                        item[1]), float(item[2])  # 节点i，节点j，距离distance
                    # （连通图）
                    if graph_type == "connect":  # 这个就是将两个节点的权重都设为1，也就相当于不要权重
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":  # 这个是有权重，下面是权重计算方法
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError(
                            "graph type is not correct (connect or distance)")
        return A

    with open(distance_file, "r") as f_d:
        f_d.readline()  # 表头，跳过第一行.第一行是表头
        reader = csv.reader(f_d)  # 读取.csv文件.
        for item in reader:  # 将一行给item组成列表
            if len(item) != 3:  # 长度应为3，不为3则数据有问题，跳过
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":  # 这个就是将两个节点的权重都设为1，也就相当于不要权重
                #A[i, j], A[j, i] = distance, distance
                A[i, j], A[j, i] = 1, 1
            elif graph_type == "distance":  # 这个是有权重，下面是权重计算方法
                # if gaussuain(distance,10) >=0.5:#高斯核构造邻接矩阵（无向图）
                #     A[i, j] = gaussuain(distance, 10)
                #     A[j, i] = gaussuain(distance, 10)
                #     print("hahaha")
                A[i, j], A[j, i] = distance, distance

            else:
                raise ValueError(
                    "graph type is not correct (connect or distance)")

    A = get_normalized_adj(A)

    return A

# 这个是载入流量数据,返回numpy的多维数组


def get_flow_data(flow_file: str) -> np.array:
    """
    :param flow_file: str, path of .npz file to save the traffic flow data
    :return:
        np.array(N, T, D)
        D为特征数实际为：(307, 16992, 1) 
    """
    # data:
    data = np.load(flow_file)

    # [N, T, D],transpose就是转置，让节点纬度在第0位，N为节点数，T为时间，D为节点特征
    flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]
    # [:, :, 0]就是只取第一个特征，[:, :, np.newaxis]就是增加一个维度，因为：一般特征比一个多，即使是一个，保持这样的习惯，便于通用的处理问题

    return flow_data  # [N, T, D]

# 获储油信息


def get_oil_data(file: str) -> np.array:   # 这个是载入流量数据,返回numpy的多维数组
    """
    :param flow_file: str, path of .npz file to save the traffic flow data
    :return:
        np.array(N, T, D)
        D为特征数实际为：(307, 16992, 1) 
    """
    # data:
    flow_data = []
    min_length = 99999
    for i in range(len(file)):
        cur_oil = pd.read_excel("oil_data/"+file[i]+".xlsx")
        min_length = min(min_length, len(cur_oil))
    for i in range(len(file)):
        cur_oil = pd.read_excel("oil_data/"+file[i]+".xlsx")
        # print(cur_oil.values[0:min_length-1, :])
        flow_data.append(cur_oil.values[0:min_length-1, :])
        plt.figure()
        plt.plot(flow_data[-1])
        plt.savefig("result/"+str(file[i])+".png")
        plt.close()
    # flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]  # [N, T, D],transpose就是转置，让节点纬度在第0位，N为节点数，T为时间，D为节点特征

    # [:, :, 0]就是只取第一个特征，[:, :, np.newaxis]就是增加一个维度，因为：一般特征比一个多，即使是一个，保持这样的习惯，便于通用的处理问题
    out = np.array(flow_data)
    # print(out.shape)
    return np.array(flow_data)[:, :, :], min_length  # [N, T, D]


# 写在这里是为了方便区分 后续放在文件头

class LoadData(Dataset):
    # 这个就是把读入的数据处理成模型需要的训练数据和测试数据，一个一个样本能读取出来
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):
        """
        :param data_path: list, ["graph file name" , "flow data file name"], path to save the data file names.
            数据的源地址
        :param num_nodes: int, number of nodes.
            你所需要的节点数目，井口数
        :param divide_days: list, [ days of train data, days of test data], list to divide the original data.
            训练数据段的和测试数据段的一个序列长list
        :param time_interval: int, time interval between two traffic data records (mins).---5 mins
            两个数据之间的差距：抽象为节点之间的平均时间间隔
        :param history_length: int, length of history data to be used.
            历史数据的一个长度
        :param train_mode: list, ["train", "test"].
        """

        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        """
        self.train_days = divide_days[0]  # 59-14 = 45, train_data
        self.test_days = divide_days[1]  # 7*2 = 14 ,test_data
        """
        self.history_length = history_length  # 30/5 = 6, 历史长度为6
        # self.time_interval = time_interval  # 5 min（每天的一个数据）

        # 这个时间数据的总偏移量
        self.one_day_length = 1  # 一整天的数据量

        # self.one_day_length = int(24 * 60 / self.time_interval) # 一整天的数据量
        '''
        #使用csv数据做一个临界矩阵
        self.graph = get_adjacent_matrix(distance_file=data_path[0], num_nodes=num_nodes)

        #此处做一个本地化处理:
        #获得真实的井数据:

        # self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1]), norm_dim=1) # self.flow_norm为归一化的基

        '''
        # 使用csv数据做一个邻接矩阵:相邻无连接关系
        self.graph = get_adjacent_matrix(
            distance_file=data_path[0], num_nodes=num_nodes, graph_type="distance")
        # print('self.graph:')
        # print(self.graph)
        self.corr = get_corr_matrix(data_path[1])
        self.dist = get_dist_matrix(
            distance_file=data_path[0], num_nodes=num_nodes)

        # print('self.corr:')
        # print(self.corr)
        # 此处做一个本地化处理:
        # 获得真实的井数据:
        # np.array(N, T, D)
        # D为特征数实际为：(307, 16992, 1)
        data_un, len = get_oil_data(data_path[1])
        print(data_un.shape)
        self.train_days = int(len*0.7)  # 59-14 = 45, train_data
        self.test_days = int(len*0.3)-1  # 7*2 = 14 ,test_data

        self.flow_norm, self.flow_data = self.pre_process_data(
            data_un, norm_dim=1)  # self.flow_norm为归一化的基

    def get_prelen(self):
        return self.test_days, self.num_nodes

    def __len__(self):  # 表示数据集的长度
        """
        :return: length of dataset (number of samples).
        """
        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length  # 　每个样本都能测试，测试样本数　＝　测试总长度
        else:
            raise ValueError(
                "train mode: [{}] is not defined".format(self.train_mode))

    # 功能是如何取每一个样本 (x, y), index = [0, L1 - 1]这个是根据数据集的长度确定的，将数据流切分成节点特征数据帧
    def __getitem__(self, index):
        """
        :param index: int, range between [0, length - 1].
        :return:
            graph: torch.tensor, [N, N].
            data_x: torch.tensor, [N, H, D].
            data_y: torch.tensor, [N, 1, D].
        """
        if self.train_mode == "train":
            index = index  # 训练集的数据是从时间０开始的，这个是每一个流量数据，要和样本（ｘ,y）区别
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length  # 有一个偏移量
        else:
            raise ValueError(
                "train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(
            self.flow_data, self.history_length, index, self.train_mode)  # 这个就是样本（ｘ,y）

        data_x = LoadData.to_tensor(data_x)  # [N, H, D] # 转换成张量
        data_y = LoadData.to_tensor(data_y).unsqueeze(
            1)  # [N, 1, D]　# 转换成张量，在时间维度上扩维

        # 组成词典返回
        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y, "corr": LoadData.to_tensor(self.corr), "dist": LoadData.to_tensor(self.dist)}

    # 静态方法，直接类名.方法名调用

    @staticmethod
    def slice_data(data, history_length, index, train_mode):  # 根据历史长度,下标来划分数据样本
        """
        :param data: np.array, normalized traffic data.
            已经归一化的交通流量数据
        :param history_length: int, length of history data to be used.
            使用历史数据的长度,历史数据指的是一次预测的一个需要的x数据
        :param index: int, index on temporal axis.

        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            H为时间下标 D为数据
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = index  # 开始下标就是时间下标本身，这个是闭区间
            end_index = index + history_length  # 结束下标,这个是开区间
        elif train_mode == "test":
            start_index = index - history_length  # 　开始下标，这个最后面贴图了，可以帮助理解
            end_index = index  # 结束下标
        else:
            raise ValueError(
                "train model {} is not defined".format(train_mode))

        data_x = data[:, start_index: end_index]  # 在切第二维，不包括end_index
        data_y = data[:, end_index]  # 把上面的end_index取上

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):  # 预处理,归一化
        """
        :param data: np.array,原始的交通流量数据
        :param norm_dim: int,归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
        :return:
            norm_base: list, [max_data, min_data], 这个是归一化的基.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = LoadData.normalize_base(
            data, norm_dim)  # 计算 normalize base
        norm_data = LoadData.normalize_data(
            norm_base[0], norm_base[1], data)  # 归一化后的流量数据

        return norm_base, norm_data  # 返回基是为了恢复数据做准备的

    @staticmethod
    def normalize_base(data, norm_dim):  # 计算归一化的基
        """
        :param data: np.array, 原始的交通流量数据
        :param norm_dim: int, normalization dimension.归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(
            data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D], keepdims=True就保持了纬度一致
        min_data = np.min(data, norm_dim, keepdims=True)

        return max_data, min_data   # 返回最大值和最小值

    @staticmethod
    def normalize_data(max_data, min_data, data):  # 计算归一化的流量数据，用的是最大值最小值归一化法
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):  # 恢复数据时使用的，为可视化比较做准备的
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data  # 这个就是原始的数据

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


def makeWelldate():
    # 控制总井组关系
    S = ["A", "A1", "A2"]

    return S

if __name__ == '__main__':

    data_sorce = []
    data_sorce.append("adjacent_matrix/adjacent_matrix.csv")
    data_sorce.append(makeWelldate())

    # data_sorce:0为连通性矩阵，1为s井组序列
    train_data = LoadData(data_path=data_sorce, num_nodes=8, divide_days=[45, 14],
                          time_interval=5, history_length=12,
                          train_mode="train")

    # print(len(train_data))

    print(train_data[0]["flow_x"].size())
    print(train_data[0]["flow_y"].size())
# 训练集样本的维度：torch.Size([307, 6, 1]  训练集标签样本的维度：torch.Size([307, 1, 1])
