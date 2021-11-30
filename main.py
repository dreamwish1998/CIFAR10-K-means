from DataReader import DataReader  # data load
from DBI import * # compute DBI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score  # compute sc
from Kmeans import KMeans  # k-means

data_reader = DataReader('./data/cifar-10-batches-py','cifar-10')  # data import
tr_data, tr_class_labels, tr_subclass_labels = data_reader.get_train_data()  # train_data train_label
tr_data_slice = tr_data[0:10000].astype("float")/255  # normalize
tr_labels_slice = tr_class_labels[0:10000]  # 1/5

k = 10  # define k

# print(tr_data_slice.shape)
# print(tr_labels_slice.shape)
# data_reader.plot_imgs(tr_data, tr_class_labels, 50, True)
kmeans = KMeans(n_clusters=k, max_iter=200)  # set the k-means model
kmeans.fit(tr_data_slice, tr_labels_slice)  # training model

# 分类中心点坐标
centers = np.array(kmeans.centroids)  # get the clusters
# 预测结果
result = kmeans.predicted_labels  # get the predict result

# print(centers.shape)
# print(len(result))
re = compute_DB_index(tr_data_slice, result, centers, k)  # the DBI metric
# re = davies_bouldin_score(tr_data_slice, result)
print("The DBI of {0} clusters: {1}".format(k, re))  # print DBI
sc = silhouette_score(tr_data_slice, result, metric='euclidean')  # the SC
print("The SC of {0} clusters: {1}".format(k, sc))  # print SC
sse = compute_SSE(tr_data_slice, result, centers, k)  # the SSE
print("The SSE of {0} clusters: {1}".format(k, sse))  # print SSE
# # data_reader.plot_imgs(kmeans.centroids, len(kmeans.centroids))
data_reader.plot_imgs(tr_data, result, k)  # output image

