import numpy as np
import matplotlib.pyplot as plt
import random

def loadData(filepath):
    '''
    加载特定格式数据
    '''
    tmp = np.loadtxt(filepath, dtype=np.double, delimiter=",")
    data_x = tmp[:,1:]
    label_y = tmp[:,:1].astype(np.int16)
    return data_x, label_y.tolist()

def dataPreprocess(data_x):
    '''
    对数据的每个维度进行均值归一化
    '''
    mean = data_x.sum(0)/len(data_x)
    mean = mean.reshape((1,len(data_x[0])))
    data_x = data_x/mean
    return data_x

def calcdistance(point1, point2):
    '''
    计算数据点间距
    '''
    dist = 0
    for i in range(len(point1)):
        dist = dist + ((point1[i]-point2[i]))**2
    return dist**0.5

def pca(X,k):#k is the components you want
    '''
    提取k个PCA特征
    '''
    #TODO 掌握PCA原理
    #mean of each feature
    n_samples, n_features = X.shape
    mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
    #normalization
    norm_X=X-mean
    #scatter matrix
    scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
    #Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
    #get new data
    data=np.dot(norm_X,np.transpose(feature))
    return data

def matPlot2D(pcaFeature, cenFeature, centers_x, k_class):
    '''
    画2维聚簇图像
    '''
    #将特征和数据分为k_class类
    x = [[] for i in range(k_class)]
    y = [[] for i in range(k_class)]
    for j in range(len(centers_x)):
        x[centers_x[j]].append(pcaFeature[j][0])
        y[centers_x[j]].append(pcaFeature[j][1])
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('cluster result')
    ax1.set_xlabel('x feature')
    ax1.set_ylabel('y feature')
    c = ['k', 'b', 'r']
    marker = ['.', 'x', '*']
    #画数据点
    for i in range(k_class):
        ax1.scatter(x[i], y[i], c=c[i], marker=marker[i])
    #画聚集中心点
    for i in range(k_class):
        ax1.scatter(cenFeature[i][0], cenFeature[i][1], c=c[i], marker='^', s=80)
    plt.show()

def matPlot(data_x, centers_x, centers, k_class):
    '''
    画聚簇图像
    '''
    #包含数据点和中心点的pca特征
    data = pca(np.concatenate((data_x, np.array(centers)), axis=0), 2)
    cenFeature = data[-k_class:, :]
    data = data[:-k_class, :]
    matPlot2D(data, cenFeature, centers_x, k_class)

def kmeansCluster(data_x, k_class):
    '''
    kmeans聚簇算法实现
    '''
    #归一化每个维度的距离，使得每个维度的间距均匀化
    data_x = dataPreprocess(data_x)
    #随机选取k个中心点
    size = len(data_x)
    #centers = np.zeros((1, k_class), np.int16)
    centers = [0 for i in range(k_class)]
    #从样本中随机选择k_class个样本点作为中心点
    for i in range(k_class):
        #centers[i] = int((np.random()*size))
        centers[i] = data_x[random.randint(0, size-1)]
    
    #从每一类中选择一个样本点作为中心点
    # centers[0] = data_x[5]
    # centers[1] = data_x[100]
    # centers[2] = data_x[166]

    #从每一类中随机选择一个样本点作为中心点
    centers[0] = data_x[random.randint(0, 58)]
    centers[1] = data_x[random.randint(59, 129)]
    centers[2] = data_x[random.randint(130, 177)]

    #距样本点最小距离的中心点
    centers_x = [0 for i in range(size)]
    #循环直到收敛
    for step in range(150):
        #统计每类的点数
        points_each_class = [[] for i in range(k_class)]

        #选取数据集最近的中心点
        #更新与中心点的距离
        for i in range(size):
            tempDist = np.zeros((k_class), np.double)
            for j in range(len(centers)):
                tempDist[j] = calcdistance(data_x[i], centers[j])
            minCenter = 0
            for j in range(len(tempDist)):
                if tempDist[j] < tempDist[minCenter]:
                    minCenter = j
            centers_x[i] = minCenter
            points_each_class[minCenter].append(data_x[i])
        #更新中心点
        for i in range(k_class):
            centers[i] = np.array(points_each_class[i]).sum(0)/len(points_each_class[i])
        
        if step%50==0:
            print("working through %d step"%step)

    matPlot(data_x, centers_x, centers, k_class)
    return centers_x

def algorithmEvaluate(convergent_center_x, label_y):
    '''
    算法评估策略
    RI: Rand Index
    '''
    #外在方法
    #Jaccard
    a = 0   # |SS|
    b = 0   # |SD|
    c = 0   # |DS|
    d = 0   # |DD|
    for i in range(len(label_y)):
        for j in range(len(convergent_center_x)):
            if(i==j):
                continue
            if(convergent_center_x[i]==convergent_center_x[j] and 
                label_y[i]==label_y[j]):
                a+=1
            elif(convergent_center_x[i]==convergent_center_x[j] and
                label_y[i]!=label_y[j]):
                b+=1
            elif(convergent_center_x[i]!=convergent_center_x[j] and
                label_y[i]==label_y[j]):
                c+=1
            else:
                d+=1
    m = len(label_y)
    RI = (a+d)/(m*(m-1))
    print("the RI rate:%f"%RI)

def KMeansCluster(filepath, kclass):
    '''
    程序执行函数
    '''
    data_x, label_y = loadData(filepath)
    center_x = kmeansCluster(data_x, kclass)
    algorithmEvaluate(center_x, label_y)


filepath = "数据集/红酒类别/wine.data"
kclass = 3
KMeansCluster(filepath, kclass)