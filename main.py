import xlrd

book = xlrd.open_workbook('C:/Users/86153/Desktop/historic/data.xlsx')

sheet1 = book.sheets()[0]
nrows = sheet1.nrows
name_of_companies = sheet1.col_values(4)
cout=0
print(name_of_companies[0])
data_read_in=[]
#print('表头为:%s\n表格总行数为:%d\n其中企业名称为：%s'%(sheet1.name,nrows,name_of_companies))
for i    in range(len(name_of_companies)):
    if (i!=0 and i<len(name_of_companies)):
        try:
            print(name_of_companies[i])
            data_read_in.append(float(name_of_companies[i]))
            cout+=1
        except:
            continue
print('接收完毕，共：%d个数据'%cout)
"""for i in range(sheet1.nrows):
    print('%s\n',sheet1.row_values(i))
    cout+=1"""
#print(cout)
"""class square:

    def __init__(self,a):
        self.b=a
    def Print(self):
        print(self.b)
    def square(self,a):
        self.b=a*a
    def add(self,a):
        self.b=self.b+a
object=square(1)
object.add(6)
object.Print()
class F1
    pass


class S1(F1):

    def show(self):
        print ('S1.show')


class S2(F1):

    def show(self):
        print ('S2.show')

def Func(obj):
    obj.show()

s1_obj = S1()
Func(s1_obj)

s2_obj = S2()
Func(s2_obj)
"""


import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 雅黑字体
"""
fig1=plt.figure(0)
plt.subplot(211)
#subplot(211)把绘图区域等分为2行*1列共两个区域，然后在区域1（上区域）中创建一个轴对象

x=np.linspace(0,20,2000)
y=np.sin(x)
z=np.sin(-x)
plt.figure(figsize=(10,6))
plt.plot(x,y,label="$y=sinx$",color="blue",linewidth=2)
plt.plot(x,z,"y--",label="$y=-sinx$")
plt.xlabel('TIME')
plt.ylabel('因变量')
plt.title("PyPlot First Example")
plt.xlabel('TIME')
plt.ylabel('因变量')

f1=plt.figure(5)#弹出对话框时的标题，如果显示的形式为弹出对话框的话
plt.subplot(221)
plt.subplot(222)
plt.subplot(212)

# subplots_adjust的操作时类似于网页css格式化中的边距处理，左边距离多少？
# 右边距离多少？这取决于你需要绘制的大小和各个模块之间的间距
x = np.linspace(0,20,2000)

y=np.sin(x)
z=np.sin(-x)

plt.figure(12)
x = np.linspace(0,20,2000)
plt.title("PyPlot First Example")
y=np.sin(x)
z=np.sin(-x)

plt.xlabel('TIME')
plt.ylabel('因变量')
plt.plot(x,y,label="$y=sinx$",color="blue",linewidth=2)
plt.xlabel('TIME')
plt.ylabel('因变量')
plt.ylim(-3,3)
plt.legend()

"""

plt.figure(figsize=(12,12))
x=range(len(name_of_companies))
y=np.sin(x)
plt.plot(x,y,"b--",label="曲线",linewidth=2)
plt.title("PyPlot First Example")
plt.ylabel('金额数值')
plt.xlabel('公司标号')
plt.legend
plt.show()




import os
import re
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark = SparkSession.builder.getOrCreate()
    df_array = []
    years = []
    air_quality_data_folder = "C:/xxx/spark/air-quality-madrid/csvs_per_year"
    for file in os.listdir(air_quality_data_folder):
        if '2018' not in file:
            year = re.findall("\d{4}", file)
            years.append(year[0])
            file_path = os.path.join(air_quality_data_folder, file)
            df = spark.read.csv(file_path, header="true")
            # print(df.columns)
            df1 = df.withColumn('yyyymm', df['date'].substr(0, 7))
            df_final = df1.filter(df1['yyyymm'].substr(0, 4) == year[0]).groupBy(df1['yyyymm']).agg({'PM10': 'avg'})
            df_array.append(df_final)

    pm10_months = [0] * 12
    # print(range(12))
    for df in df_array:
        for i in range(12):
            rows = df.filter(df['yyyymm'].contains('-'+str(i+1).zfill(2))).first()
            # print(rows[1])
            pm10_months[i] += (rows[1]/12)

    years.sort()
    print(years[0] + ' - ' + years[len(years)-1] + '年，每月平均PM10统计')
    m_index = 1
    for data in pm10_months:
        print(str(m_index).zfill(2) + '月份: ' + '||' * round(data))
        m_index += 1


"""#condig: utf - 8
import torch as th
import numpy as np


class GM():

    def __init__(self):
        # 判断是否可用 gpu 编程 , 大量级计算使用GPU
        self._is_gpu = False  # th.cuda.is_available()

    def fit(self, dt: list or np.ndarray):
        self._df: th.Tensor = th.from_numpy(np.array(dt, dtype=np.float32))

        if self._is_gpu:
            self._df.cuda()

        self._n: int = len(self._df)

        self._x, self._max_value = self._sigmod(self._df)

        z: th.Tensor = self._next_to_mean(th.cumsum(self._x, dim=0))

        self.coef: th.Tensor = self._coefficient(self._x, z)

        del z

        self._x0: th.Tensor = self._x[0]

        self._pre: th.Tensor = self._pred()

    # 归一化
    def _sigmod(self, x: th.Tensor):
        _maxv: th.Tensor = th.max(x)
        return th.div(x, _maxv), _maxv

    # 计算紧邻均值数列
    def _next_to_mean(self, x_1: th.Tensor):

        z: th.Tensor = th.zeros(self._n - 1)
        if self._is_gpu:
            z.cuda()

        for i in range(1, self._n):  # 下标从0开始，取不到最大值
            z[i - 1] = 0.5 * x_1[i] + 0.5 * x_1[i - 1]

        return z

    # 计算系数 a,b
    def _coefficient(self, x: th.Tensor, z: th.Tensor):

        B: th.Tensor = th.stack((-1 * z, th.ones(self._n - 1)), dim=1)

        Y: th.Tensor = th.tensor(x[1:], dtype=th.float32).reshape((-1, 1))

        if self._is_gpu:
            B.cuda()
            Y.cuda()

        # 返回的是a和b的向量转置，第一个是a 第二个是b；
        return th.matmul(th.matmul(th.inverse(th.matmul(B.t(), B)), B.t()), Y)

    def _pred(self, start: int = 1, end: int = 0):

        les: int = self._n + end

        resut: th.Tensor = th.zeros(les)

        if self._is_gpu:
            resut.cuda()

        resut[0] = self._x0

        for i in range(start, les):
            resut[i] = (self._x0 - (self.coef[1] / self.coef[0])) * \
                       (1 - th.exp(self.coef[0])) * th.exp(-1 * self.coef[0] * (i))
        del les
        return resut

    # 计算绝对误差
    def confidence(self):
        return round((th.sum(th.abs(th.div((self._x - self._pre), self._x))) / self._n).item(), 4)

    # 预测个数，默认个数大于等于0，
    def predict(self, m: int = 1, decimals: int = 4):

        y_pred: th.Tensor = th.mul(self._pre, self._max_value)

        y_pred_ = th.zeros(1)

        if m < 0:
            return "预测个数需大于等于0"
        elif m > 0:
            y_pred_: th.Tensor = self._pred(self._n, m)[-m:].mul(self._max_value)
        else:
            if self._is_gpu:
                return list(map(lambda _: round(_, decimals), y_pred.cpu().numpy().tolist()))
            else:
                return list(map(lambda _: round(_, decimals), y_pred.numpy().tolist()))

        # cat 拼接 0 x水平拼接，1y垂直拼接
        result: th.Tensor = th.cat((y_pred, y_pred_), dim=0)

        del y_pred, y_pred_

        if self._is_gpu:
            return list(map(lambda _: round(_, decimals), result.cpu().numpy().tolist()))

        return list(map(lambda _: round(_, decimals), result.numpy().tolist()))


if __name__ == "__main__":
    ls = np.arange(50, 100, 2)
    print(type(ls))
    # ls = list()
    gm = GM()
    gm.fit(ls)
    print(gm.confidence())
    print(ls)
    print(gm.predict(m=2))"""



"""import pandas as pd
from pandas import DataFrame, Series
data = DataFrame({'name':['yang', 'jian', 'yj'], 'age':[23, 34, 22], 'gender':['male', 'male', 'female']})
#data数据
'''
In[182]: data
Out[182]: 
   age  gender  name
0   23    male  yang
1   34    male  jian
2   22  female    yj
'''
#删除gender列，不改变原来的data数据，返回删除后的新表data_2。axis为1表示删除列，0表示删除行。inplace为True表示直接对原表修改。
#data_2 = data.drop('gender', axis=1, inplace=False)
'''
In[184]: data_2
Out[184]: 
   age  name
0   23  yang
1   34  jian
2   22    yj
'''
#改变某一列的位置。如：先删除gender列，然后在原表data中第0列插入被删掉的列。
print(data)
print(data.insert(0, '性别',None))
#pop返回删除的列，插入到第0列，并取新名为'性别'
print(data)
'''
In[185]: data
Out[186]: 
       性别  age  name
0    male   23  yang
1    male   34  jian
2  female   22    yj
'''
#直接在原数据上删除列
del data['性别']
'''
In[188]: data
Out[188]: 
   age  name
0   23  yang
1   34  jian
2   22    yj
'''"""


"""多项式回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import genfromtxt
import xlrd
book=xlrd.open_workbook('C:/Users/86153/Desktop/MMM_sec_time/data/Model_3_indicators.xlsx')
data_read_in=[]
sheet_1=book.sheets()[0]
print(type(sheet_1))
for i in range(sheet_1.nrows):
    try:
        raw_values=sheet_1.row_values(i)
        data_read_in.append(int(raw_values[9]))
    except:
        continue
data_read_in.pop(0)
y=np.arange(len(data_read_in))
print(data_read_in)


data_read_in=np.array(data_read_in)
print('done\n',data_read_in)
print(type(sheet_1))
data_read_in=np.delete(data_read_in,0,0)
print('done\n',data_read_in)


plt.scatter(data_read_in, y)
#y_=np.arange(0,20000,20000/14)
# 一维变二维
data_read_in_1 = np.array(data_read_in)
data_read_in_2 = data_read_in_1[:, np.newaxis]
y_2 = y[:, np.newaxis]

#y_2_=y_[:, np.newaxis]
print(data_read_in_2)
# 创建并拟合模型
model = LinearRegression()
model.fit(data_read_in_2, y_2)

plt.plot(data_read_in_2, model.predict(data_read_in_2), 'r')

#非线性回归
# 定义多项式回归, degree的值可以调节多项式的特征
poly_model = PolynomialFeatures(degree=8)

# 特征处理
data_read_in_multi = poly_model.fit_transform(x_train)

# 定义回归模型
reg = LinearRegression()
reg.fit(x_train, y_train_)

plt.plot(x_train,reg.predict(data_read_in_multi), 'g')
plt.show()
"""

#数据拟合
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    x_points = [0, 1, 2, 3, 4, 5]
    y_points = [1,2,4,8,16,32]  # 实际函数关系式为：y=x^2

    xnew = np.linspace(min(x_points), max(x_points), 100)  # 新制作100个x值。(等差、list[]形式存储)

    print(type(xnew))
    tck = interpolate.splrep(x_points, y_points)

    ynew = interpolate.splev(xnew, tck)  # 通过拟合的曲线，计算每一个输入值。(100个结果，list[]形式存储)

    plt.scatter(x_points[:], y_points[:], 25, "red")  # 绘制散点
    plt.plot(xnew, ynew)  # 绘制拟合曲线图
    plt.show()
    print('start to compute')
    for i in x_future:


    interpolate.splev(x, tck)

    return
print(f(10))
"""import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import genfromtxt
from pylab import mpl
import math
TRAIN_TOTAL=66
PREDICT_END=100
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 雅黑字体
#从excel中读入数据
import xlrd
data_read_in = []

book = xlrd.open_workbook('C:/Users/86153/Desktop/MMM_sec_time/data/Model_3_indicators.xlsx')
data_read_in = []
sheet_1 = book.sheets()[0]
print(type(sheet_1),'have_read_in_data')
i=0
for i in range(TRAIN_TOTAL+2):
    try:
        raw_values = sheet_1.row_values(i)
        use_value=float(raw_values[9])
        data_read_in.append(use_value)

    except:
        continue
print(data_read_in,len(data_read_in))
def f(x1):
    return 0
#加载数据函数
def load_data():

    x1_train = np.linspace(0,TRAIN_TOTAL,TRAIN_TOTAL+1)
    print(x1_train)
    #x1_train_list=list((x1_train))
    #print(x1_train_list)
    data_train = np.array([[x1,data_read_in[int(x1)]] for x1 in x1_train])
    print(data_train)
    print(type(data_train))

    #x1_future_list=list(x1_future)
    x1_future=np.linspace(TRAIN_TOTAL+1,PREDICT_END,PREDICT_END-TRAIN_TOTAL+1)
    data_future = np.array([[x1,0] for x1 in x1_future])
    print(data_future)
    print(type(data_future))
    return data_train, data_future

train,  future = load_data()
x_train, y_train = train[:,:1], train[:,1]
#print(x_train)
#print(y_train)
x_future, y_future = future[:,:1] ,future[:,1]#train[:,:1],即train[:,0:1],train[:,0]

plt.figure(figsize=(8,4))
plt.plot(x_train[:,0],y_train,'r--',label='train')

plt.plot(x_future[:,0],y_future,'b--',label='future')
plt.legend()
plt.show()


###########2.回归部分##########
def try_different_method(model,x_train_=x_train):
    model.fit(x_train_,y_train)
   # score = model.score(x_test, y_test)
    future =model.predict(x_future)
    plt.figure(figsize=(10,5))
    plt.plot(x_train_[:,0],y_train,'r--',label='pre value')

    #plt.plot(np.arange(len(result),len(result)+len(future)),y_future,'yo-',label='future true value')

    plt.plot(np.arange(len(data_read_in),len(data_read_in)+len(future)),future,'bo-',label='future predict value')
    print(y_future,'\n')
    print(future,'\n')
    plt.title('use method of %s'%model)
    plt.legend()
    plt.show()
###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
####3.10ARD贝叶斯ARD回归
model_ARDRegression = linear_model.ARDRegression()
####3.11BayesianRidge贝叶斯岭回归
model_BayesianRidge = linear_model.BayesianRidge()
####3.12TheilSen泰尔森估算
model_TheilSenRegressor = linear_model.TheilSenRegressor()
####3.13RANSAC随机抽样一致性算法
model_RANSACRegressor = linear_model.RANSACRegressor()


"""

###########4.具体方法调用部分##########
print('决策树回归结果')
try_different_method(model_DecisionTreeRegressor)

print('线性回归结果')
try_different_method(model_LinearRegression)


#非线性多项式回归

def try_so_different_method(new_model):
    poly = PolynomialFeatures(degree=8, include_bias=False)  # the bias is avoiding the need to intercept
    x_train_spi = poly.fit_transform(x_train)
    new_model.fit(x_train_spi, y_train)
    x_new = poly.fit_transform(x_future)
    y_new_prediction = new_model.predict(x_new)
    # plotting
    y_prediction = new_model.predict(x_train_spi)  # this predicts y

    plt.scatter(x_train, y_train)
    plt.plot(x_train_spi[:, 0], y_prediction, 'r')
    plt.plot(x_new[:, 0], y_new_prediction, 'g--')
    plt.legend(['Predicted line', 'Observed data'])
    plt.title('use method of polynomial regression')
    plt.legend()
    plt.show()
    return
print('非线性回归结果\n')
try_so_different_method(LinearRegression())




#SVM回归结果
try_different_method(model_SVR)

#KNN回归结果
try_different_method(model_KNeighborsRegressor)

#贝叶斯ARD回归结果
try_different_method(model_ARDRegression)
#随机森林回归结果

try_different_method(model_RandomForestRegressor)

#Adaboost回归结果
try_different_method(model_AdaBoostRegressor)

#GBRT回归结果
try_different_method(model_GradientBoostingRegressor)

#Bagging回归结果
try_different_method(model_BaggingRegressor)

#极端随机树回归结果
try_different_method(model_ExtraTreeRegressor)



#贝叶斯岭回归结果
try_different_method(model_BayesianRidge)

#泰尔森估算回归结果
try_different_method(model_TheilSenRegressor)

#随机抽样一致性算法
try_different_method(model_RANSACRegressor)
"""
from scipy import interpolate

def f_interpolate():
    y_future = []
    xnew = np.linspace(0, TRAIN_TOTAL, 100)  # 新制作100个x值。(等差、list[]形式存储)
    x_points=np.linspace(0,TRAIN_TOTAL,TRAIN_TOTAL+1)
    y_points=data_read_in
    tck = interpolate.splrep(x_points, y_points)

    ynew = interpolate.splev(xnew, tck)  # 通过拟合的曲线，计算每一个输入值。(100个结果，list[]形式存储)

    #plt.scatter(x_points[:], y_points[:], 25, "red")  # 绘制散点
    #plt.plot(xnew, ynew)  # 绘制拟合曲线图
    #plt.show()
    print('start to compute')
    for i in x_future[:,0]:

        y_future.append(interpolate.splev(i, tck))
    plt.plot(x_train[:, 0], y_train, 'r--', label='train')

    plt.plot(x_future[:, 0], y_future, 'bo', label='future')

    print(y_future, '\n')
    print(type(y_future) ,'\n')
    plt.title('use method of interpolate')
    plt.legend()
    plt.show()

    return

#插值拟合
f_interpolate()


"""
#非线性回归
# 定义多项式回归, degree的值可以调节多项式的特征
print('非线性多项式回归预测')
y_train_=train[:,1:]
poly_model = PolynomialFeatures(degree=8)

# 特征处理
data_read_in_multi = poly_model.fit_transform(x_train)

# 定义回归模型
reg = LinearRegression()
reg.fit(x_train, y_train_)

plt.plot(x_train,reg.predict(data_read_in_multi), 'g')
plt.title('use method of polynomial regression')
plt.legend()
plt.show()
"""

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# =============================================================================
# 指数函数拟合
# =============================================================================
def func_exponent(x, a, b ):
    return a*np.exp(-b * x)+1/150

# =============================================================================
# 幂指数函数拟合
# =============================================================================
def func_base(x, a, b ):
    return x**a +b

# =============================================================================
# 多项式函数拟合
# =============================================================================
def func_polynomial(x, a, b, c ):
    return a*x**2+ b*x +c

def func_logarithm(x,a,b):
    return np.log(x)*a + b

from scipy.optimize import curve_fit

def try_exp_or_base_method(func_input):
    xdata=x_train[:, 0]
    ydata=y_train
    xfuture=x_future[:, 0]


    # 函数拟合
    popt, pcov = curve_fit(func_input, xdata, ydata)  # 若为指数拟合，指数popt数组中，三个值分别是待求参数a,b,c
    # 原值拟合
    y_pred = [func_input(i, popt[0], popt[1]) for i in xdata]
    #未来值预测
    y_to_be_pre=[func_input(i,popt[0],popt[1]) for i in xfuture]
    # 画图
    plt.plot(xdata, y_pred, 'b')
    plt.plot(x_future, y_to_be_pre, 'r--')
    print('curve_fit预测结果：\n',y_to_be_pre)
    plt.title('use method of curve_fit,exactly %s'%func_input.__name__)
    plt.legend()
    plt.show()

    """# 输出R方,用于评定预测结果
    from sklearn.metrics import r2_score
    r2 = r2_score(ydata, y_pred)
    print('指数函数拟合R方为:', r2)
  """


try_exp_or_base_method(func_base)
try_exp_or_base_method(func_logarithm)
#用多项式要改变方程"""
