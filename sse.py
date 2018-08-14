import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import  math
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
path = 'D:\Pythontest\用户分析\电商消费行为\data\\'

data = pd.read_csv(open(path+'zscoreddata.csv'))
if __name__ == '__main__':
    def distEclud(vecA, vecB):
        """
        计算两个向量的欧式距离的平方，并返回
        """
        return np.sum(np.power(vecA - vecB, 2))


    def test_Kmeans_nclusters(data_train):
        """
        计算不同的k值时，SSE的大小变化
        """
        data_train = data_train.values
        nums = range(2, 10)
        SSE = []
        for num in nums:
            sse = 0
            kmodel = KMeans(n_clusters=num, n_jobs=4)
            kmodel.fit(data_train)
            # 簇中心
            cluster_ceter_list = kmodel.cluster_centers_
            # 个样本属于的簇序号列表
            cluster_list = kmodel.labels_.tolist()
            for index in range(len(data)):
                cluster_num = cluster_list[index]
                sse += distEclud(data_train[index, :], cluster_ceter_list[cluster_num])
            print("簇数是", num, "时； SSE是", sse)
            SSE.append(sse)
        return nums, SSE

    nums, SSE = test_Kmeans_nclusters(data)


    def draw_k_line(n, s):
        # 画图，通过观察SSE与k的取值尝试找出合适的k值
        # 中文和负号的正常显示
        plt.rcParams['font.sans-serif'] = 'SimHei'
        plt.rcParams['font.size'] = 12.0
        plt.rcParams['axes.unicode_minus'] = False
        # 使用ggplot的绘图风格
        plt.style.use('ggplot')
        ## 绘图观测SSE与簇个数的关系
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(n, s, marker="+")
        ax.set_xlabel("n_clusters", fontsize=18)
        ax.set_ylabel("SSE", fontsize=18)
        fig.suptitle("KMeans", fontsize=20)
        plt.show()

    draw_k_line(nums, SSE)