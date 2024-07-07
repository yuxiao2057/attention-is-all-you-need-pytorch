import numpy as np
from numpy.random import randn

#1、定义输入数据a
#a表示有32个向量的输入数据，每个向量是一个256维的列向量 
d = 256
n = 32
a = np.random.rand(d,n)            #(256, 32)

#2、计算数据数据对应的Q,K,V
Wq = np.random.rand(d,d)
Wk = np.random.rand(d,d)
Wv = np.random.rand(d,d)

#输入a是32*256的
#wq是256*256的
Q = np.dot(Wq,a)      #(256, 32)
K = np.dot(Wk,a)
V = np.dot(Wv,a)

#2、计算数据数据对应的Q,K,V
Wq = np.random.rand(d,d)
Wk = np.random.rand(d,d)
Wv = np.random.rand(d,d)

#3、计算注意力得分score，记为：A
#计算公式为；A = softmax(K^T*Q)

#定义sofmax函数
def soft_max(z):
    z = np.clip(z,100,-100)  #归一化至[-100,100],防止softmax时数据溢出
    t = np.exp(z)
    res = np.exp(z) / np.sum(t, axis=1) 
    return res

A = soft_max(np.dot(K.T,Q))        #(32, 32)

#4、计算self-attention的输出数据：O
# O = V * A
# O的维数和输入数据a保持一致
O = np.dot(V,A)                   #(256, 32)