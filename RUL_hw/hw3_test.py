import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.ma.core import indices
from tensorboard.plugins.scalar.summary import scalar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Flatten, LSTM, Dense, Dropout,BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

data_X_train=np.genfromtxt("data\\train_FD001.txt",delimiter=' ')
RUL_test=np.genfromtxt("data\\RUL_FD001.txt",delimiter=' ')
data_X_test=np.genfromtxt("data\\test_FD001.txt",delimiter=' ')

#生成训练数据的RUL
data_Y_train = np.zeros(data_X_train.shape[0])
for i in range(100):
    indices=np.where(data_X_train[:,0]==i+1)[0]
    RUL_max=data_X_train[indices[-1],1]
    array_range = np.arange(RUL_max, 0, -1)  # 从 RUL_max 到 1 的倒序数组
    data_Y_train[indices] = array_range[:len(indices)] -1

#生成测试数据的RUL
data_Y_test=np.zeros(data_X_test.shape[0])
for i in range(RUL_test.shape[0]):
    indices=np.where(data_X_test[:,0]==i+1)[0]
    array_range = np.arange(RUL_test[i]+data_X_test[indices[-1],1], 0, -1)
    data_Y_test[indices] = array_range[:len(indices)] -1

std1=np.std(data_X_train[:,2:],axis=0)
print(f"标准化之前的标准差：")
print(std1)
avg1=np.mean(data_X_train[:,2:],axis=0)
print(f"标准化前的平均值：")
print(avg1)

# 标准化
scaler = StandardScaler()
data_X_train_standardized = scaler.fit_transform(data_X_train[:, 2:])
data_X_test_standardized = scaler.transform(data_X_test[:, 2:])
std2=np.std(data_X_train_standardized,axis=0)
avg2=np.mean(data_X_train_standardized,axis=0)
print(f"标准化后的标准差：")
print(std2)
print(f"标准化后的平均值：")
print(avg2)
#展示部分原始训练数据
sample_indices = [0, 1, 2, 3]  
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(15, 20))
# 遍历每个特征
for feature_idx in range(24):
    # 找到每个样本的时间步进
    time_steps = []
    for sample_idx in sample_indices:
        indices = np.where(data_X_train[:, 0] == sample_idx + 1)[0]
        time_steps.append(data_X_train[indices, 1])
    # 绘制每个样本的特征
    for i, sample_idx in enumerate(sample_indices):
        indices = np.where(data_X_train[:, 0] == sample_idx + 1)[0]
        axes[feature_idx // 4, feature_idx % 4].plot(time_steps[i], data_X_train[indices, feature_idx + 2],
                                                     label=f'Sample {sample_idx + 1}')
    # 设置子图的标题和标签
    axes[feature_idx // 4, feature_idx % 4].set_title(f'Feature {feature_idx + 1}')
    axes[feature_idx // 4, feature_idx % 4].set_xlabel('RUL')
    axes[feature_idx // 4, feature_idx % 4].set_ylabel('Value')
    axes[feature_idx // 4, feature_idx % 4].legend()
plt.tight_layout()
plt.show()


#展示部分标准化后的训练数据
sample_indices = [0, 1, 2, 3]
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(15, 20))
# 遍历每个特征
for feature_idx in range(24):
    # 找到每个样本的时间步进
    time_steps = []
    for sample_idx in sample_indices:
        indices = np.where(data_X_train[:, 0] == sample_idx + 1)[0]
        time_steps.append(data_X_train[indices, 1])

    # 绘制每个样本的特征
    for i, sample_idx in enumerate(sample_indices):
        indices = np.where(data_X_train[:, 0] == sample_idx + 1)[0]
        axes[feature_idx // 4, feature_idx % 4].plot(time_steps[i], data_X_train_standardized[indices, feature_idx],
                                                     label=f'Sample {sample_idx + 1}')
    # 设置子图的标题和标签
    axes[feature_idx // 4, feature_idx % 4].set_title(f'Feature {feature_idx + 1}')
    axes[feature_idx // 4, feature_idx % 4].set_xlabel('RUL')
    axes[feature_idx // 4, feature_idx % 4].set_ylabel('Value')
    axes[feature_idx // 4, feature_idx % 4].legend()
plt.tight_layout()
plt.show()


# 设置PCA的维度
n_components = 8
# 进行PCA降维
pca = PCA(n_components=n_components)
data_X_train_standardized_pca = pca.fit_transform(data_X_train_standardized)
data_X_test_standardized_pca = pca.transform(data_X_test_standardized)
explained_variance_ratio = pca.explained_variance_ratio_
# 打印每个主成分的信息占比
for i, ratio in enumerate(explained_variance_ratio):
    print(f"主成分 {i+1} 的信息占比: {ratio:.4f}")
# 计算并展示降维后数据保留的总信息占比
total_explained_variance_ratio = sum(explained_variance_ratio)
print(f"降维后数据保留的总信息占比: {total_explained_variance_ratio:.4f}")

# 保留前两列数据
data_X_train_first_two_columns = data_X_train[:, :2]
data_X_test_first_two_columns = data_X_test[:, :2]

# 将PCA处理后的数据与保留的前两列数据合并
data_X_train_pca = np.hstack((data_X_train_first_two_columns, data_X_train_standardized_pca))
data_X_test_pca = np.hstack((data_X_test_first_two_columns, data_X_test_standardized_pca))
sample_indices = [0, 1, 2, 3]
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 20))
# 遍历每个特征
for feature_idx in range(8):
    # 找到每个样本的时间步进
    time_steps = []
    for sample_idx in sample_indices:
        indices = np.where(data_X_train[:, 0] == sample_idx + 1)[0]
        time_steps.append(data_X_train[indices, 1])
    # 绘制每个样本的特征
    for i, sample_idx in enumerate(sample_indices):
        indices = np.where(data_X_train[:, 0] == sample_idx + 1)[0]
        axes[feature_idx // 4, feature_idx % 4].plot(time_steps[i], data_X_train_pca[indices, feature_idx + 2],
                                                     label=f'Sample {sample_idx + 1}')
    # 设置子图的标题和标签
    axes[feature_idx // 4, feature_idx % 4].set_title(f'Feature {feature_idx + 1}')
    axes[feature_idx // 4, feature_idx % 4].set_xlabel('RUL')
    axes[feature_idx // 4, feature_idx % 4].set_ylabel('Value')
    axes[feature_idx // 4, feature_idx % 4].legend()
plt.tight_layout()
plt.show()



max_RUL_train=int(max(data_Y_train)+1)
max_RUL_test=int(max(data_Y_test)+1)

def cause_padding(data_x,data_y,max_time_step):
    num_features=data_x.shape[1]
    padding_array=np.zeros((max_time_step,num_features))
    data_padded_x=np.zeros((100,max_time_step,num_features))
    data_padded_y=np.zeros((100,max_time_step))
    for i in range(100):
        indices=np.where(data_x[:,0]==i+1)[0]
        row,col=data_x[indices].shape
        padding_array[:row,:col]=data_x[indices]
        data_padded_x[i,:,:]=padding_array
        row_y=data_y[indices].shape[0]
        data_padded_y[i,:row_y]=data_y[indices]
    return data_padded_x,data_padded_y

data_x_train,data_y_train=cause_padding(data_X_train_pca,data_Y_train,max_RUL_train)
data_x_test,data_y_test=cause_padding(data_X_test_pca,data_Y_test,max_RUL_train)

# 定义1D CNN模型
def build_1d_cnn_model(input_shape, num_classes):
    model = Sequential()
    # 添加卷积层
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='linear'))
    return model


input_shape = (data_x_train.shape[1], data_x_train.shape[2])
print(f"Input shape: {input_shape}")

model = build_1d_cnn_model(input_shape, max_RUL_train)
# 编译
model.compile(optimizer=Adam(), loss='mse')
# 打印模型结构
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# 训练
history = model.fit(data_x_train, data_y_train, epochs=100, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping])
# 绘制训练损失和验证损失
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
# 测试
predictions = model.predict(data_x_test)
print(predictions.shape)
prediction_RUL=np.zeros(predictions.shape[0])
for i in range(100):
    indices = np.where(data_x_test[i, :, 0] == i + 1)[0]
    #print(indices[-1])
    prediction_RUL[i]=predictions[i,indices[-1]]
id=np.arange(1,101)
plt.plot(id, prediction_RUL,label='prediction')
plt.plot(id,RUL_test,label='real')
plt.legend()
plt.show()
