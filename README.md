# 基于Sympy实现的梯度下降算法
- 使用方法更简单,易于理解的实现方式
- 但相比起矩阵运算效率可能会稍低...日后可能会更新更高效率的实现方式
# 使用方法
   ```
       # 测试数据,请保证Y处于最后一列
       data = {
           'x': [
           0,1,2,3,4,5,6,7,8,9,
           10,11,12,13,14,15,16,17,18,19],
           'y': [
           3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
           11, 13, 13, 16, 17, 18, 17, 19, 21]
           }
       # 将数据以DataFrame形式储存(格式为: x1,x2,...,xn,y)
       frame1 = DataFrame(data)
       # 获取线性回归结果(Θ0,Θ1,Θ2,....Θn)
       result = GradientDescent(frame1,a=0.01)
       print("=============== result as below ====================")
       theta = result.theta
       print(theta)
       
       # 将结果转为可视化的图片
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
   ```
# 其他
请联系 cyx2706@163.com