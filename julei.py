
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

def kmeans(data,k,max_iter=100,tol=1e-4):
    centroids=initialize_centroids(data,k)
    clusters=np.zeros(len(data))
    for i in range(max_iter):
        clusters=assign_clusters(data,centroids)
        new_centroids=update_centroids(data,clusters,k)

        if np.all(np.abs(new_centroids-centroids)<tol):
            print('达到最小容差值，停止迭代')
            centroids = new_centroids
            return clusters,centroids
        centroids=new_centroids
    print('达到最大迭代次数，停止迭代')
    return clusters,centroids

# 示例数据
data = np.random.rand(200, 2)
k = 4  # 类别数量

clusters=np.zeros(len(data))
centroids=np.zeros(k)
clusters, centroids = kmeans(data, k)

# 可视化结果
plt.scatter(data[:, 0], data[:, 1],c=clusters, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
#plt.clabel('Feature 3')
plt.show()


