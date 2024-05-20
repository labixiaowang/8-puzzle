import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

# 设置中文字符支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 从Excel加载数据
data = pd.read_excel(r"C:\Users\28135\Desktop\k-mearns.xlsx")

def safe_log(x):
    return x.apply(lambda val: 0 if val == 0 else np.log(val))
data['出货量对数'] = safe_log(data['出货量'])
sales_data = data['出货量对数'].values.reshape(-1, 1)
meandistortions = []
# k值取1-10
K = range(1,10)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sales_data)
    meandistortions.append(sum(np.min(cdist(sales_data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / sales_data.shape[0])
# 画肘部图
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('平均距离')
plt.title('肘部图')
plt.show()


# 3.k-means模型
kmeans = KMeans(n_clusters=4)  # 选择聚类的簇数，根据你的需求进行调整
# 4.拟合
kmeans.fit(sales_data)

# 5. 获取聚类中心
centroids = kmeans.cluster_centers_
print("聚类中心：")
print(centroids)

# 6. 获取每个样本的聚类标签
labels = kmeans.labels_
data['类别'] = labels  # 将聚类标签添加到数据中
data.to_excel('k-means聚类(取对数).xlsx',index=False)