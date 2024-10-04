
import numpy as np
import matplotlib.pyplot as plt
def initialize_centroids(data,k):
    indices=np.random.choice(len(data),k,replace=False)
    centroids=data[indices]
    return centroids

def assign_clusters(data,centroids):
    k=len(centroids)
    clusters=np.zeros(len(data))
    for i in range(len(data)):
        distances=np.linalg.norm(data[i]-centroids,axis=1)
        clusters[i]=np.argmin(distances)
    return clusters

def update_centroids(data,clusters,k):
    centroids=np.zeros((k,data.shape[1]))
    for i in range(k):
        centroids[i]=data[clusters==i].mean(axis=0)
    return centroids


import numpy as np


def kmeans(data, k, max_iter=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    clusters = np.zeros(len(data))

    for i in range(max_iter):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        if np.all(np.abs(new_centroids - centroids) < tol):
            centroids = new_centroids
            break
        centroids = new_centroids

    # 计算SSE
    sse = 0
    for j in range(k):
        points = data[clusters == j]
        sse += np.sum((points - centroids[j]) ** 2)

    return clusters, centroids, sse


def find_optimal_k(data,max_iter=100,tol=1e-4):
    sse=np.zeros(10)
    for i in range(10):
        clusters,centroids,sse[i]=kmeans(data,(i+1),max_iter,tol)
    diff=np.diff(sse)
    second_deri=np.diff(sse)
    optimal_k=np.argmax(diff)
    return optimal_k
# 示例数据
data = np.random.rand(200, 2)
# 类别数量

sse=0
optimal_k= find_optimal_k(data)
clusters=np.zeros(len(data))
centroids=np.zeros(optimal_k)
clusters,centroids,sse=kmeans(data,optimal_k)


# 可视化结果
plt.scatter(data[:, 0], data[:, 1],c=clusters, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
#plt.clabel('Feature 3')
plt.show()


