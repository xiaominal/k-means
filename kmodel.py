
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
    def draw_radar_map(data_t):
        kmodel = KMeans(n_clusters=5, n_jobs=4)
        kmodel.fit(data)
        r1 = pd.Series(kmodel.labels_).value_counts()
        r2 = pd.DataFrame(kmodel.cluster_centers_)
        # 所有簇中心坐标值中最大值和最小值
        max = r2.values.max()
        min = r2.values.min()
        r = pd.concat([r2, r1], axis=1)
        r.columns = list(data.columns) + [u'类别数目']
        r_USER = pd.concat([data, pd.Series(kmodel.labels_, index=data.index)], axis=1)  # 详细输出每个样本对应的类别
        r_USER.columns = list(data.columns) + [u'聚类类别']  # 重命名表头
        r_USER.to_csv(path + 'cluster_data5.csv')  # 保存结果

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        center_num = r.values
        feature = ['ZM', 'ZF', 'ZR']
        N = len(feature)
        for i, v in enumerate(center_num):
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            # 设置雷达图的角度，用于平分切开一个圆面
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            # print(angles)
            # 为了使雷达图一圈封闭起来，需要下面的步骤
            center = np.concatenate((v[:-1], [v[0]]))
            angles = np.concatenate((angles, [angles[0]]))
            # print('b',angles)
            # 绘制折线图
            ax.plot(angles, center, 'o-', linewidth=2, label="第%d簇人群,%d人" % (i + 1, v[-1]))
            # 填充颜色
            ax.fill(angles, center, alpha=0.25)
            # 添加每个特征的标签
            ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize=15)
            # 设置雷达图的范围
            ax.set_ylim(min - 0.1, max + 0.1)
            # 添加标题
            plt.title('客户特征分析图', fontsize=20)
            # 添加网格线
            ax.grid(True)
            # 设置图例
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncol=1, fancybox=True, shadow=True)
        plt.show()


    draw_radar_map(data)