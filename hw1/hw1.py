import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize

iris=np.genfromtxt("C:\\Users\\86134\\PycharmProjects\\hw1\\.venv\\share\\iris.txt",delimiter=None)
Y=iris[:,-1]
X=iris[:,0:-1]
#(a)
print("the number of features %d"%(X.shape[1]))
print("the number of data points %d "%(X.shape[0]))
#(b)
for i in range(X.shape[1]):
    plt.hist(x=X[:,i],bins=30,edgecolor='white',density=True)
    plt.show()
#(c)

    avg=X[:,i].mean()
    print("the average value of features %d"%(i)+" is %f"%avg)
#(d)

    var=np.var(X[:,i])
    std=np.std(X[:,i])
    print("the variance of features %d" % (i) + " is %f" %var)
    print("the standard deviation of features %d" % (i) + " is %f" %std)
#(e)
      #初始化一个存储标化后的数据的数组
    data_normalized=np.zeros((X.shape[0],X.shape[1]))
    data_normalized=(X[:,i]-avg)/std
#（f）
for j in range(X.shape[1]-1):
    plt.scatter(X[:,0],X[:,j+1],c=Y,cmap='viridis')
    plt.title("feature (1,%d)"%(j+2))
    plt.show()