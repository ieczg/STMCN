# @Time    : 2020/8/16 
# @Author  : LeronQ
# @github  : https://github.com/LeronQ


#  utils.py

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import h5py
from datetime import datetime
import csv

from numpy.core.numeric import count_nonzero 

class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target - output) / (target + 5)) # 加５是因为target有可能为0，当然只要不太大，加几都行

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target, output)
        mape = Evaluation.mape_(target, output)
        rmse = Evaluation.rmse_(target, output)

        return mae, mape, rmse


def visualize_result(h5_file, para,nodes_id, time_se,wellList, visualize_file,my_net):
    file_obj = h5py.File(h5_file, "r") # 获得文件对象，这个文件对象有两个keys："predict"和"target"
    prediction = file_obj["predict"][:][:, :, 0]  # [N, T],切片，最后一维取第0列，所以变成二维了，要是[:, :, :1]那么维度不会缩减
    target = file_obj["target"][:][:, :, 0]  # [N, T],同上
    file_obj.close()

    headers = ['Time',"model","para","well","MAE","RMSE","MAPE"]    
    dt=datetime.now()
    with open("result/"+"Mean_Square_Error.csv",'a',newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for i in range(nodes_id):
            plot_prediction = prediction[i][time_se[0]: time_se[1]]  # [T1]，将指定节点的，指定时间的数据拿出来
            plot_target = target[i][time_se[0]: time_se[1]]  # [T1]，同上

            #复原预测结果
            rmse= np.sqrt((np.sum(((np.squeeze(plot_prediction) - np.squeeze(plot_target))**2 )))/len(plot_prediction))
            #平均绝对值误差
            mae = (np.sum(abs(np.squeeze(plot_prediction) - np.squeeze(plot_target))))/len(plot_prediction)
            #平均绝对百分比误差
            mape=(np.sum(abs((np.squeeze(plot_prediction) - np.squeeze(plot_target))/np.squeeze(plot_target))))/len(plot_prediction)

            rows=[dt.strftime( '%Y-%m-%d %H:%M:%S %f' ),str(my_net)]#一次写入列表的一行，矩阵形式的一项就是一行
            rows.append(para)
            rows.append(wellList[i])
            rows.append("{:.2f}".format((mae)))
            rows.append("{:.2f}".format((rmse)))
            rows.append("{:.2f}".format((mape)))
            f_csv.writerow(rows)
            plt.figure()
            plt.grid(True, linestyle="-.", linewidth=0.5)
            plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction, ls="-", marker=" ", color="r")
            plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target, ls="-", marker=" ", color="b")

            plt.legend(["prediction", "target"], loc="upper right")

            plt.axis([0, time_se[1] - time_se[0],
                np.min(np.array([np.min(plot_prediction), np.min(plot_target)])),
                np.max(np.array([np.max(plot_prediction), np.max(plot_target)]))])

            plt.savefig(visualize_file+wellList[i] + ".png")
            plt.close()
         
        