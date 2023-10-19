# @Time    : 2020/8/25
# @Author  : LeronQ
# @github  : https://github.com/LeronQ

# traffic_prediction.py
import torch
import os
import time
import h5py
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Oil_dateset import LoadData  # 这个就是上一小节处理数据自己写的的类，封装在traffic_dataset.py文件中
from utils import Evaluation  # 三种评价指标以及可视化类
from utils import visualize_result
import Oil_dateset
import warnings
import drs2
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU

historyLength = 12
batchSize = 128  #
# batchSize = 64
# batchSize = 32
isTrain = True
#isTrain = False
LR = 0.005
# LR = 0.01
Epoch = 50  # 训练的次数
# Epoch = 100  # 训练的次数
# Epoch = 150  # 训练的次数
# Epoch = 200  # 训练的次数
# Epoch = 1
jk_mode = "max"
# jk_mode = "cat"
num_layers = 6  # gat层数
num_layer = 3  # 时空模块个数


def train(wellList):

    num_node = len(wellList)

    # 第一步：准备数据（上一节已经准备好了，这里只是调用而已，链接在最开头）
    data_sorce = []
    data_sorce.append("adjacent_matrix/adjacent_matrix.csv")
    data_sorce.append(wellList)

    train_data = LoadData(data_path=data_sorce, num_nodes=num_node, divide_days=[45, 14],
                          time_interval=5, history_length=historyLength,
                          train_mode="train")
    print("+++++++++++++++++++++++++++++++++++" + str(train_data.__len__()))
    # 将num_workers改为0
    train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True,
                              num_workers=0, drop_last=True)  # num_workers是加载数据（batch）的线程数目


    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # 定义设备

    # 第二步：定义模型（这里其实只是加载模型，关于模型的定义在下面单独写了，先假设已经写好）初始化
    my_net = drs2.STGNSNet(device, adj=torch.tensor(train_data.graph, dtype=torch.float32), corr=torch.tensor(train_data.corr, dtype=torch.float32), dist=torch.tensor(train_data.dist, dtype=torch.float32),
                           in_channels=1, batch_size=batchSize, embed_size=64, time_num=12, num_layers=num_layer, gat_layers=num_layers, mode=jk_mode, T_dim=historyLength, output_T_dim=1, heads=1, dropout=0.05, forward_expansion=4)


    # my_net = my_net.to(device)  # 模型送入设备
    my_net = my_net.cuda()  # 模型送入设备
    # 第三步：定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方损失函数

    # 没写学习率，表示使用的是默认的，也就是lr=1e-4
    optimizer = optim.Adam(params=my_net.parameters(),
                           lr=LR)

    ExpLR = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.97)
    # 第四步：训练+测试
    # Train model

    my_net.train()  # 打开训练模式
    lossdrow = []
    cur_len = 0
    for epoch in range(Epoch):
        # print("=================================================" + str(epoch))
        cur_len = cur_len+1
        epoch_loss = 0.0
        count = 0
        start_time = time.time()
        # ["graph/corr": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]],一次把一个batch的训练数据取出来
        for data in train_loader:
            my_net.zero_grad()  # 梯度清零
            count += 1
            # 这行代码就是将加载的ChebNet应用到具体的数据上
            # predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D],由于标签flow_y在cpu中，所以最后的预测值要放回到cpu中
            # [B, N, 1, D],由于标签flow_y在cpu中，所以最后的预测值要放回到cpu中
            predict_value = my_net(data, device).to(torch.device("cpu"))

            # 计算损失，切记这个loss不是标量
            loss = criterion(predict_value, data["flow_y"])

            epoch_loss += loss.item()  # 这里是把一个epoch的损失都加起来，最后再除训练数据长度，用平均loss来表示

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            ExpLR.step()  # 更新学习率
        end_time = time.time()
        epoch_loss_draw = 1000 * epoch_loss / len(train_data)
        lossdrow.append(epoch_loss_draw)
        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, epoch_loss_draw,
                                                                          (end_time - start_time) / 60))
        if (epoch % 1 == 0):
            # 保存模式：
            # 保存整个网络
            torch.save({'model': my_net.state_dict()},
                       "result/parameter.pth")

        # print(my_net.state_dict())

    # 保存模式：
    # 保存整个网络
    torch.save({'model': my_net.state_dict()},
               "result/parameter.pth")

    plt.figure()
    plt.plot(lossdrow)
    plt.savefig("result/loss.png")
    plt.close()
    # Test Model
    # 对于测试:
    # 第一、除了计算loss之外，还需要可视化一下预测的结果（定性分析）
    # 第二、对于预测的结果这里我使用了 MAE, MAPE, and RMSE 这三种评价标准来评估（定量分析）


def testModel(wellList):

    num_node = len(wellList)

    # 第一步：准备数据（上一节已经准备好了，这里只是调用而已，链接在最开头）
    data_sorce = []
    data_sorce.append("adjacent_matrix/adjacent_matrix.csv")
    data_sorce.append(wellList)

    train_data = LoadData(data_path=data_sorce, num_nodes=num_node, divide_days=[45, 14],
                          time_interval=5, history_length=historyLength,
                          train_mode="train")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # 定义设备

    my_net = drs2.STGNSNet(device, adj=torch.tensor(train_data.graph, dtype=torch.float32), corr=torch.tensor(train_data.corr, dtype=torch.float32), dist=torch.tensor(train_data.dist, dtype=torch.float32),
                           in_channels=1, batch_size=batchSize, embed_size=64, time_num=12, num_layers=num_layer, gat_layers=num_layers, mode=jk_mode, T_dim=historyLength, output_T_dim=1, heads=1, dropout=0.05, forward_expansion=4)
    if torch.cuda.is_available():
        pass
    state_dict = torch.load("result/parameter.pth")
    my_net.load_state_dict(state_dict['model'])


    my_net.eval()  # 打开测试模式

    test_data = LoadData(data_path=data_sorce, num_nodes=num_node, divide_days=[45, 14],
                         time_interval=5, history_length=historyLength,
                         train_mode="test")

    prelen, num_nodes = test_data.get_prelen()

    # 将num_workers改为0
    test_loader = DataLoader(
        test_data, batch_size=batchSize, shuffle=False, num_workers=2, drop_last=True)

    prelen = test_loader.__len__()*batchSize
    criterion = nn.MSELoss()  # 均方损失函数
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
    my_net = my_net.to(device)  # 模型送入设备

    with torch.no_grad():  # 关闭梯度
        MAE, MAPE, RMSE = [], [], []  # 定义三种指标的列表
        Target = np.zeros([num_node, 1, 1])  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充
        Predict = np.zeros_like(Target)  # [N, T, D],T=1 # 预测数据的维度

        total_loss = 0.0
        for data in test_loader:  # 一次把一个batch的测试数据取出来

            # 下面得到的预测结果实际上是归一化的结果，有一个问题是我们这里使用的三种评价标准以及可视化结果要用的是逆归一化的数据
            # predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D]，B是batch_size, N是节点数量,1是时间T=1, D是节点的流量特征
            # [B, N, 1, D]，B是batch_size, N是节点数量,1是时间T=1, D是节点的流量特征
            predict_value = my_net(data, device).to(torch.device("cpu"))

            loss = criterion(predict_value, data["flow_y"])  # 使用MSE计算loss

            total_loss += loss.item()  # 所有的batch的loss累加
            # 下面实际上是把预测值和目标值的batch放到第二维的时间维度，这是因为在测试数据的时候对样本没有shuffle，
            # 所以每一个batch取出来的数据就是按时间顺序来的，因此放到第二维来表示时间是合理的.
            predict_value = predict_value.transpose(0, 2).squeeze(
                0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
            target_value = data["flow_y"].transpose(0, 2).squeeze(
                0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]

            performance, data_to_save = compute_performance(
                predict_value, target_value, test_loader)  # 计算模型的性能，返回评价结果和恢复好的数据

            # 下面这个是每一个batch取出的数据，按batch这个维度进行串联，最后就得到了整个时间的数据，也就是
            # [N, T, D] = [N, T1+T2+..., D]
            Predict = np.concatenate([Predict, data_to_save[0]], axis=1)
            Target = np.concatenate([Target, data_to_save[1]], axis=1)

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])
            # 每个batch的误差
            print("Test Loss: {:02.4f}".format(
                1000 * total_loss / len(test_data)))

    # 三种指标取平均,np.mean()求取均值
    print("Performance:  MAE {:2.2f}    MAPE {:2.2f}%    RMSE {:2.2f}".format(
        np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)))

    # 将第0行的0删除，因为开始定义的时候用0填充，但是时间是从1开始的
    Predict = np.delete(Predict, 0, axis=1)
    Target = np.delete(Target, 0, axis=1)

    result_file = "result/result.h5"
    # h5数据集存放预测结果与真实结果 用以结果可视化
    file_obj = h5py.File(result_file, "w")  # 将预测值和目标值保存到文件中，因为要多次可视化看看结果

    file_obj["predict"] = Predict  # [N, T, D]
    file_obj["target"] = Target  # [N, T, D]

    return prelen, num_nodes, type(my_net)


def compute_performance(prediction, target, data):  # 计算模型性能
    # 下面的try和except实际上在做这样一件事：当训练+测试模型的时候，数据肯定是经过dataloader的，所以直接赋值就可以了
    # 但是如果将训练好的模型保存下来，然后测试，那么数据就没有经过dataloader，是dataloader型的，需要转换成dataset型。
    try:
        dataset = data.dataset  # 数据为dataloader型，通过它下面的属性.dataset类变成dataset型数据
    except:
        dataset = data  # 数据为dataset型，直接赋值

    # 下面就是对预测和目标数据进行逆归一化，recover_data()函数在上一小节的数据处理中
    #  flow_norm为归一化的基，flow_norm[0]为最大值，flow_norm[1]为最小值
    # prediction.numpy()和target.numpy()是需要逆归一化的数据，转换成numpy型是因为 recover_data()函数中的数据都是numpy型，保持一致
    prediction = LoadData.recover_data(
        dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_data(
        dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())

    # 对三种评价指标写了一个类，这个类封装在另一个文件中，在后面
    mae, mape, rmse = Evaluation.total(
        target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）


if __name__ == '__main__':
    wellLists = Oil_dateset.makeWelldate()
    type_net = ""
    if (isTrain):
        train(wellLists)
        prelen, num_nodes, type_net = testModel(wellLists)
    else:
        prelen, num_nodes, type_net = testModel(wellLists)
    print(prelen)
    # 可视化，在下面的 Evaluation()类中，这里是对应的GAT算法运行的结果，进行可视化
    # 如果要对GCN或者chebnet进行可视化，只需要在第45行，注释修改下对应的算法即可
    Evaluation()
    # 通过result可视化图谱
    visualize_result(h5_file="result/result.h5",
                     para="LR ="+str(LR)+"  "+"Epoch ="+str(Epoch)+"  "+"jk_mode ="+str(jk_mode),
                    nodes_id=num_nodes, time_se=[0, prelen], wellList=wellLists, 
                    visualize_file="result/gat_node_", my_net=type_net)  # 是节点的时间范围
        