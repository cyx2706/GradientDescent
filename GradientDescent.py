from pandas import Series,DataFrame
import numpy as np
from sympy import *
class GradientDescent:
    """
	基于sympy简单实现的梯度下降算法
	:author cyx2706@163.com
	"""
    #求出的theta列表(numpy形式)
    theta = None

    # 步长
    a = 0

    #梯度的阈值(当梯度极度靠近0时,即代表已到达最低点,不再往下迭代,避免死循环)
    threshold = 1e-5

    #样本数量
    m = 0

    #特征数(维数)
    n = 0

    #损失函数对象
    jfunc = None

    def __init__(self, dataframe,a = 0.01,threshold = 1e-5):
        """
        :parameter dataframe 数据,格式x1,...,xn,y
        :parameter a 步长(即学习率)
        :parameter threshold 梯度的阈值,建议改值设为极度靠近0的数(当梯度极度靠近0时,即代表已到达最低点,不再往下迭代,避免死循环)
        """
        self.n = len(dataframe.columns.values[0:-1])
        self.m = len(dataframe)
        self.a = a
        self.threshold = threshold

        # 初始化Θ,默认起始点为(1,1,....1)
        _theta_vec = []
        for i in range(self.n+1):
            _theta_vec.append(1)
        self.theta = np.array(_theta_vec)

        # 定义损失函数
        THETA, self.jfunc = self.j(dataframe)

        # 重复迭代以达到最低点
        do_while = False # 实现do...while
        gradient_dist = 0 # 梯度的长度(标量)
        while do_while == False or gradient_dist > self.threshold:
            do_while = True
            # 对损失函数进行微分,获取微分后的表达式, 即Δj(Θ)
            gradient_f =  self.delta_j(THETA)
            # 代入当前的Θ向量到Δj(Θ)中
            THETA_VALUES = {}
            for i in range(len(self.theta)):
                THETA_VALUES["Θ{0}".format(i)] = self.theta[i]
            gradient = self.a * np.array( list(map(lambda f: f.evalf(subs = THETA_VALUES) , gradient_f )) )
            # 更新Θ向量
            self.theta = self.theta -  gradient
            # 记录当次Θ向量向梯度方向移动的长度,当长度低于阈值时将不再迭代
            gradient_dist = np.sqrt( float(np.sum(np.square( gradient ) ) ) )
        return

    def h(self,THETA,X):
        """
        构造 h(x) = Θ0 + Θ1*x1 + ... +  Θn*xn
        :parameter THETA Θ symbols 列表
        :parameter X X值的列表(即一行)
        """
        hfunc = THETA[0]
        for i in range(self.n):
            hfunc += THETA[i+1]*X[i]
        return hfunc

    def j(self,dataframe):
        """
        构造损失函数的表达式
        :parameter dataframe 数据集
        """
        n = self.n
        # 构建Θ向量
        _theta_symbol_names = ["Θ0"]
        for i in range(n):
            _theta_symbol_names.append("Θ{0}".format(i+1))
            
        THETA = symbols(" ".join(_theta_symbol_names))

        #构建损失函数
        m = self.m# 样本个数m
        #j(Θ) = 1/2*m*( ∑(h(Xj)-Y)**2 )
        jfunc  = 0
        for row_index in dataframe.index:
            X = dataframe.loc[row_index].values[0:-1]
            Y = dataframe.loc[row_index].values[-1]
            jfunc += (1/(2*m))*( self.h(THETA , X) - Y )**2
        return [THETA, jfunc]

    def delta_j(self, THETA):
        """
        构造梯度Δj(Θ)的表达式
        :parameter THETA Θ symbols 列表
        """
        delta_jfunc = []
        for i in range(len(THETA)):
            #解出代价函数的梯度，也就是分别对各个变量进行微分
            delta_jfunc.append(diff(self.jfunc,THETA[i]))
        return delta_jfunc

"""
测试
"""
if __name__ == "__main__":
    data = {
        'x': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
        'y': [
        3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
        11, 13, 13, 16, 17, 18, 17, 19, 21
    ]}
    frame1 = DataFrame(data)
    theta = GradientDescent(frame1,a=0.01).theta
    print("=============== result as below ====================")
    print(theta)

    import matplotlib.pyplot as plt
    x = np.arange(0, 21, 0.1)
    y = theta[0] + theta[1]*x
    plt.title("result")
    plt.plot(x, y)
    plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],[
        3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
        11, 13, 13, 16, 17, 18, 17, 19, 21
    ],'ro')
    plt.show()

