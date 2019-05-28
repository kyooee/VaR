# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:15:52 2019

@author: jy_hbcf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import  stats
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

# 读取数据
data = pd.read_csv('D:/R/VaR/000041.csv',encoding='utf-8')
#data.head()
ts=pd.Series(list(data['NAV']),index=list(data['TradingDate']))
ts.index=pd.to_datetime(ts.index)#将日净值序列变为时间序列
ts=ts.sort_index()#基金日序列按时间顺序排序
plt.plot(ts)
plt.show()
r=np.log(ts)-np.log(ts.shift(1))#计算基金日收益率
r=r.dropna()
plt.plot(r)
plt.show()

R=r[465:]*100
trainseries=R[:1385]
trainset=np.array(trainseries)
trainset=trainset.reshape(trainset.shape[0],1)
testseries=R[1385:]
testset=np.array(testseries)
testset=testset.reshape(testset.shape[0],1)
[len(trainseries),len(testseries)]

#创建数据集
def create_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 5
trainX, trainY = create_dataset(trainset, look_back)
volatility1=[np.var(trainX[i]) for i in range(len(trainX))] 
trainX=np.hstack((trainX,np.array(volatility1).reshape(len(trainX),-1)))
train=pd.DataFrame(np.hstack((trainY.reshape(len(trainX),-1),trainX)),columns=['r','r_1','r_2','r_3','r_4','r_5','vol'])

testX, testY = create_dataset(testset, look_back)
volatility2=[np.var(testX[i]) for i in range(len(testX))] 
testX=np.hstack((testX,np.array(volatility2).reshape(len(testX),-1)))
test=pd.DataFrame(np.hstack((testY.reshape(len(testX),-1),testX)),columns=['r','r_1','r_2','r_3','r_4','r_5','vol'])


#分位数回归
import statsmodels.formula.api as smf

#训练集
mod = smf.quantreg('r~r_1+r_2+r_3+r_4+r_5+vol',train)     
quantiles = [0.1,0.05,0.01]
models = []
params = []
for qt in quantiles:
    print(qt)
    res = mod.fit(q = qt )
    models.append(res)
    params.append([qt, res.params['Intercept'], res.params['r_1'],res.params['r_2'],res.params['r_3'],res.params['r_4'],res.params['r_5'],res.params['vol']] )
    params = pd.DataFrame(data = params, columns = ['qt','intercept','r_1_coef','r_2_coef','r_3_coef','r_4_coef','r_5_coef','vol_coef'])


r_pred0 = models[0].params['Intercept'] + models[0].params['r_1'] * train['r_1']+ models[0].params['r_2'] * train['r_2']+ models[0].params['r_3'] * train['r_3']+models[0].params['r_4'] * train['r_4']+ models[0].params['r_5'] * train['r_5']+ models[0].params['vol'] * train['vol']
r_pred0=pd.Series(np.array(r_pred0),index=list(trainseries.index[6:]))

r_pred1 = models[1].params['Intercept'] + models[1].params['r_1'] * train['r_1']+ models[1].params['r_2'] * train['r_2']+ models[1].params['r_3'] * train['r_3']+models[1].params['r_4'] * train['r_4']+ models[1].params['r_5'] * train['r_5']+ models[1].params['vol'] * train['vol']
r_pred1=pd.Series(np.array(r_pred1),index=list(trainseries.index[6:]))

r_pred2 = models[2].params['Intercept'] + models[2].params['r_1'] * train['r_1']+ models[2].params['r_2'] * train['r_2']+ models[2].params['r_3'] * train['r_3']+models[2].params['r_4'] * train['r_4']+ models[2].params['r_5'] * train['r_5']+ models[2].params['vol'] * train['vol']
r_pred2=pd.Series(np.array(r_pred2),index=list(trainseries.index[6:]))

VaR1=[r_pred0,r_pred1,r_pred2]


#测试集
tr_pred0 = models[0].params['Intercept'] + models[0].params['r_1'] * test['r_1']+ models[0].params['r_2'] * test['r_2']+ models[0].params['r_3'] * test['r_3']+models[0].params['r_4'] * test['r_4']+ models[0].params['r_5'] * test['r_5']+ models[0].params['vol'] * test['vol']
tr_pred0=pd.Series(np.array(tr_pred0),index=list(testseries.index[6:]))

tr_pred1 = models[1].params['Intercept'] + models[1].params['r_1'] * test['r_1']+ models[1].params['r_2'] * test['r_2']+ models[1].params['r_3'] * test['r_3']+models[1].params['r_4'] * test['r_4']+ models[1].params['r_5'] * test['r_5']+ models[1].params['vol'] * test['vol']
tr_pred1=pd.Series(np.array(tr_pred1),index=list(testseries.index[6:]))

tr_pred2 = models[2].params['Intercept'] + models[2].params['r_1'] * test['r_1']+ models[2].params['r_2'] * test['r_2']+ models[2].params['r_3'] * test['r_3']+models[2].params['r_4'] * test['r_4']+ models[2].params['r_5'] * test['r_5']+ models[2].params['vol'] * test['vol']
tr_pred2=pd.Series(np.array(tr_pred2),index=list(testseries.index[6:]))

tVaR1=[tr_pred0,tr_pred1,tr_pred2]



# 绘制训练集与测试集的VaR
plt.figure(figsize=(12,6))
plt.plot(R,color='black',linewidth=0.5, label='Rate of Return')
plt.plot(VaR1[0], 'b--',linewidth=0.5, label='Q Reg : 0.01')
plt.plot(VaR1[1], 'g--',linewidth=0.5, label='Q Reg : 0.05')
plt.plot(VaR1[2], 'r--',linewidth=0.5, label='Q Reg : 0.1')
plt.plot(tVaR1[0], 'b--',linewidth=0.5,)
plt.plot(tVaR1[1], 'g--',linewidth=0.5)
plt.plot(tVaR1[2], 'r--',linewidth=0.5)
plt.vlines(R.index[1385], -18, 18, colors = "c", linestyles = "dashed")
plt.legend()
plt.show()


#分位数回归森林
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=20000, min_samples_leaf=1,random_state=3,n_jobs=-1)
rf.fit(trainX, trainY)
def rf_quantile(model, X, q=0.05):
    rf_preds = []
    for estimator in model.estimators_:
        rf_preds.append(estimator.predict(X))
    rf_preds = np.array(rf_preds).transpose()  # One row per record.
    return np.percentile(rf_preds, q*100, axis=1)

'''
r_pred= rf_quantile(rf, trainX, q=0.01)
r_pred=pd.Series(r_pred,index=list(trainseries.index[6:]))
L1=trainseries[6:]-r_pred
n=len(L1)#回测历史数据总数
L2=[j for j in L1 if j<0]
m=len(L2)#失败次数
p=m/n
p
'''
#训练集
q=[0.031,0.010,0.00025]
VaR2=[]
for i in q:
    r_pred= rf_quantile(rf, trainX, q=i)
    r_pred=pd.Series(r_pred,index=list(trainseries.index[6:]))
    VaR2.append(r_pred)
 
    
#测试集
tVaR2=[]
for i in q:
    tr_pred= rf_quantile(rf, testX, q=i)
    tr_pred=pd.Series(tr_pred,index=list(testseries.index[6:]))
    tVaR2.append(tr_pred)


# 绘制训练集与测试集的VaR
plt.figure(figsize=(12,6))
plt.plot(R,color='black',linewidth=0.5, label='Rate of Return')
plt.plot(VaR2[0], 'b--',linewidth=0.5, label='Q Reg : 0.01')
plt.plot(VaR2[1], 'g--',linewidth=0.5, label='Q Reg : 0.05')
plt.plot(VaR2[2], 'r--',linewidth=0.5, label='Q Reg : 0.1')
plt.plot(tVaR2[0], 'b--',linewidth=0.5,)
plt.plot(tVaR2[1], 'g--',linewidth=0.5)
plt.plot(tVaR2[2], 'r--',linewidth=0.5)
plt.vlines(R.index[1385], -8, 8, colors = "c", linestyles = "dashed")
plt.legend()
plt.show()


#经验模式分解
decomposer=EMD(R)
imfs = decomposer.decompose()
#绘制分解图
def plot_imfs(signal, imfs, time_samples=None, fignum=None, show=True):

    is_bivariate = np.any(np.iscomplex(signal))
    if time_samples is None:
        time_samples = np.arange(signal.shape[0])

    n_imfs = imfs.shape[0]

    fig = plt.figure(num=fignum,figsize=(15,20))
    axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))

    # Plot original signal
    ax = plt.subplot(n_imfs + 1, 1, 1)
    if is_bivariate:
        ax.plot(time_samples, np.real(signal), 'b')
        ax.plot(time_samples, np.imag(signal), 'k--')
    else:
        ax.plot(time_samples, signal)
    ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('Signal')
    ax.set_title('Empirical Mode Decomposition')

    # Plot the IMFs
    for i in range(n_imfs - 1):
        ax = plt.subplot(n_imfs + 1, 1, i + 2)
        if is_bivariate:
            ax.plot(time_samples, np.real(imfs[i]), 'b')
            ax.plot(time_samples, np.imag(imfs[i]), 'k--')
        else:
            ax.plot(time_samples, imfs[i])
        ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        ax.grid(False)
        ax.set_ylabel('imf' + str(i + 1))

    # Plot the residue
    ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
    if is_bivariate:
        ax.plot(time_samples, np.real(imfs[-1]), 'r')
        ax.plot(time_samples, np.imag(imfs[-1]), 'r--')
    else:
        ax.plot(time_samples, imfs[-1])
    ax.axis('tight')
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('res.')

    if show:  # pragma: no cover
        plt.show()
    return fig

plot_imfs(R,imfs,R.index)
#描述统计
arr = np.vstack((R,imfs))
df_arr=pd.DataFrame(arr.T,columns=['r','imf_1','imf_2','imf_3','imf_4','imf_5','imf_6','imf_7','imf_8','res'])
df_arr.describe()
#拟相关系数
L=[]
for i in range(len(imfs)):
    Re=np.dot(R,imfs[i])/(np.linalg.norm(r)*np.linalg.norm(imfs[i]))
    L.append(Re)
print(L)
#方差贡献 
TS=0
for i in range(len(imfs)):
     TS=TS+np.var(imfs[i])
print(TS)
L=[]
for i in range(len(imfs)):
    STDR=np.var(imfs[i])/TS
    L.append(STDR)
print(L)
#平均周期
from scipy.signal import argrelextrema
m0=len(argrelextrema(np.array(r), np.greater)[0])
c0=len(r)/m0
print(c0)
L=[]
for i in range(len(imfs)):
    m=len(argrelextrema(np.array(imfs[i]), np.greater)[0])
    c=len(r)/m
    L.append(c)
print(L)

#IMF分量重构   
random=imfs[0]+imfs[1]
detail=imfs[4]+imfs[5]+imfs[6]+imfs[7]+imfs[3]
trend=imfs[8]

arr2=np.vstack((R,random))
arr2=np.vstack((arr2,detail))
arr2=np.vstack((arr2,trend))
df_arr2=pd.DataFrame(arr2.T,columns=['r','random','detail','trend'])
df_arr2.describe()
#统计量
R1=np.dot(R,random)/(np.linalg.norm(R)*np.linalg.norm(random))
R2=np.dot(R,detail)/(np.linalg.norm(R)*np.linalg.norm(detail))
R3=np.dot(R,detail)/(np.linalg.norm(R)*np.linalg.norm(detail))
print([R1,R2,R3])
STDR1=np.var(random)/(np.var(random)+np.var(detail)+np.var(trend))
STDR2=np.var(detail)/(np.var(random)+np.var(detail)+np.var(trend))
STDR3=np.var(trend)/(np.var(random)+np.var(detail)+np.var(trend))
print([STDR1,STDR2,STDR3])
c0=len(R)/len(argrelextrema(np.array(R), np.greater)[0])
c1=len(R)/len(argrelextrema(np.array(random), np.greater)[0])
c2=len(R)/len(argrelextrema(np.array(detail), np.greater)[0])
print([c0,c1,c2])

plt.figure(figsize=(12,10))
plt.subplot(411)
plt.plot(np.array(R),color='black',linewidth=0.5, label='Rate of Return')
plt.legend()
plt.subplot(412)
plt.plot(random, 'b--',linewidth=0.5, label='random')
plt.legend()
plt.subplot(413)
plt.plot(detail, 'g--',linewidth=0.5, label='detail')
plt.legend()
plt.subplot(414)
plt.plot(trend, 'r--',linewidth=0.5, label='trend')
plt.legend()
plt.show()

# EMD-QRF
## 划分测试集，训练集
train1=trend[:1385]
test1=trend[1385:]
trainX1, trainY1 = create_dataset(train1.reshape(len(train1),-1), look_back=5)
volatility1=[np.var(trainX1[i]) for i in range(len(trainX1))] 
trainX1=np.hstack((trainX1,np.array(volatility1).reshape(len(trainX1),-1)))
testX1, testY1 = create_dataset(test1.reshape(len(test1),-1), look_back=5)
volatility1=[np.var(testX1[i]) for i in range(len(testX1))] 
testX1=np.hstack((testX1,np.array(volatility1).reshape(len(testX1),-1)))

train2=detail[:1385]
test2=detail[1385:]
trainX2, trainY2 = create_dataset(train1.reshape(len(train2),-1), look_back=5)
volatility2=[np.var(trainX2[i]) for i in range(len(trainX2))] 
trainX2=np.hstack((trainX2,np.array(volatility2).reshape(len(trainX2),-1)))
testX2, testY2 = create_dataset(test2.reshape(len(test2),-1), look_back=5)
volatility2=[np.var(testX2[i]) for i in range(len(testX2))] 
testX2=np.hstack((testX2,np.array(volatility2).reshape(len(testX2),-1)))

train3=random[:1385]
test3=random[1385:]
trainX3, trainY3 = create_dataset(train3.reshape(len(train3),-1), look_back=5)
volatility3=[np.var(trainX3[i]) for i in range(len(trainX3))] 
trainX3=np.hstack((trainX3,np.array(volatility3).reshape(len(trainX3),-1)))
testX3, testY3 = create_dataset(test3.reshape(len(test3),-1), look_back=5)
volatility3=[np.var(testX3[i]) for i in range(len(testX3))] 
testX3=np.hstack((testX3,np.array(volatility3).reshape(len(testX3),-1)))

#趋势分量
pred1=trainX1[:,0]
tpred1=testX1[:0]
#低频分量
pred2=trainX2[:,0]
tpred2=testX2[:0]
#高频分量

def Monte(data,q):
    mean=[np.var(data[i]) for i in range(len(data))] 
    vol=[np.var(data[i]) for i in range(len(data))] 
    d=pd.DataFrame()
    d.mean=mean
    d.vol=vol
    var=[]
    for i in range(len(data)):
        MC=1000
        rr=list(np.random.normal(loc=mean[i],scale=vol[i]**0.5,size = (MC,1)))
        var.append(np.percentile(rr,0.05))
    return var

rf3 = RandomForestRegressor(n_estimators=20000, min_samples_leaf=1)
rf3.fit(trainX3, trainY3)

VaR3=[]
for i in q:
    r_pred1=trainX1[:,0]
    r_pred2=trainX2[:,0]
    r_pred3= rf_quantile(rf3, trainX3, q=i)
    r_pred=r_pred1+r_pred2+r_pred3
    r_pred=pd.Series(r_pred,index=list(trainseries.index[6:]))
    VaR3.append(r_pred)

tVaR3=[]
for i in q:
    tr_pred1=testX1[:,0]
    tr_pred2=testX2[:,0]
    tr_pred3= rf_quantile(rf3, testX3, q=i)
    tr_pred=tr_pred1+tr_pred2+tr_pred3    
    tr_pred=pd.Series(tr_pred,index=list(testseries.index[6:]))
    tVaR3.append(tr_pred)

# 绘制训练集与测试集的VaR
plt.figure(figsize=(12,6))
plt.plot(R,color='black',linewidth=0.5, label='Rate of Return')
plt.plot(VaR3[0], 'b--',linewidth=0.5, label='Q Reg : 0.01')
plt.plot(VaR3[1], 'g--',linewidth=0.5, label='Q Reg : 0.05')
plt.plot(VaR3[2], 'r--',linewidth=0.5, label='Q Reg : 0.1')
plt.plot(tVaR3[0], 'b--',linewidth=0.5,)
plt.plot(tVaR3[1], 'g--',linewidth=0.5)
plt.plot(tVaR3[2], 'r--',linewidth=0.5)
plt.vlines(r.index[2082], -8, 8, colors = "c", linestyles = "dashed")
plt.legend()
plt.show()


#Kupiec检验
alpha=[0.1,0.05,0.01]
M1=[]
P1=[]
LR1=[]
for i in [0,1,2]:
    L1=trainseries[6:]-VaR1[i]

    n=len(L1)#回测历史数据总数
    L2=[j for j in L1 if j<0]
    m=len(L2)#失败次数
    p=m/n
    lr=-2*((n-m)*math.log(1-alpha[i]) +m* math.log(alpha[i])-(n-m)*math.log(1-p)-m*math.log(p))
    M1.append(m)
    P1.append(p)
    LR1.append(lr)
print(M1)
print(P1)
print(LR1)

tM1=[]
tP1=[]
tLR1=[]
for i in [0,1,2]:
    L1=testseries[6:]-tVaR1[i]

    n=len(L1)#回测历史数据总数
    L2=[j for j in L1 if j<0]
    m=len(L2)#失败次数
    p=m/n
    lr=-2*((n-m)*math.log(1-alpha[i]) +m* math.log(alpha[i])-(n-m)*math.log(1-p)-m*math.log(p))
    tM1.append(m)
    tP1.append(p)
    tLR1.append(lr)
print(tM1)
print(tP1)
print(tLR1)

M2=[]
P2=[]
LR2=[]
for i in [0,1,2]:
    L1=trainseries[6:]-VaR2[i]
    n=len(L1)#回测历史数据总数
    L2=[j for j in L1 if j<0]
    m=len(L2)#失败次数
    p=m/n
    lr=-2*((n-m)*math.log(1-alpha[i]) +m* math.log(alpha[i])-(n-m)*math.log(1-p)-m*math.log(p))
    M2.append(m)
    P2.append(p)
    LR2.append(lr)
print(M2)
print(P2)
print(LR2)



tM2=[]
tP2=[]
tLR2=[]
for i in [0,1,2]:
    L1=testseries[6:]-tVaR2[i]
    n=len(L1)#回测历史数据总数
    L2=[j for j in L1 if j<0]
    m=len(L2)#失败次数
    p=m/n
    lr=-2*((n-m)*math.log(1-alpha[i]) +m* math.log(alpha[i])-(n-m)*math.log(1-p)-m*math.log(p))
#    lr2=(n-m)*math.log(1-(p))-m*math.log(p)
#    lr=-2*lr1+2*lr2
    tM2.append(m)
    tP2.append(p)
    tLR2.append(lr)
print(tM2)
print(tP2)
print(tLR2)


#Kupiec检验
alpha=[0.1,0.05,0.01]
M3=[]
P3=[]
LR3=[]
for i in [0,1,2]:
    L1=trainseries[6:]-VaR3[i]
    n=len(L1)#回测历史数据总数
    L2=[j for j in L1 if j<0]
    m=len(L2)#失败次数
    p=m/n
    lr1=(n-m)*math.log(1-alpha[i]) +m* math.log(alpha[i])
    lr2=(n-m)*math.log(1-(m/n))+m*math.log(m/n)
    lr=-2*lr1+2*lr2
    M3.append(m)
    P3.append(p)
    LR3.append(lr)
print(M3)
print(P3)
print(LR3)

tM3=[]
tP3=[]
tLR3=[]

for i in [0,1,2]:
    L1=testseries[6:]-tVaR3[i]
    n=len(L1)#回测历史数据总数
    L2=[j for j in L1 if j<0]
    m=len(L2)#失败次数
    p=m/n
    lr1=(n-m)*math.log(1-alpha[i]) +m* math.log(alpha[i])
    lr2=(n-m)*math.log(1-(m/n))+m*math.log(m/n)
    lr=-2*lr1+2*lr2
    tM3.append(m)
    tP3.append(p)
    tLR3.append(lr)
print(tM3)
print(tP3)
print(tLR3)



