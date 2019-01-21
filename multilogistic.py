import numpy as np
import tensorflow as tf
import sklearn.utils import shuffle

#필요한　변수　설정
M=2
K=3
n=100
N=n*K
#샘플데이터군을　생성
X1=np.random.rand(n,M)+np.array([0,10])
X2=np.random.rand(n,M)+np.array([5,5])
X3=np.random.rand(n,M)+np.array([10,0])
Y1=np.array([[1,0,0] for i in range(n)])
Y2=np.array([[0,1,0] for i in range(n)])
Y3=np.array([[0,0,1] for i in range(n)])

X=np.concatenate((X1,X2,X3),axis=0)
Y=np.concatenate((Y1,Y2,Y3),axis=0)

#모델을　정의:이진분류에　사용되었던　시그모이드를　소프트맥스로만　변경해주면　된다．
W=tf.Variable(tf.zeros([M,K]))
b=tf.Variable(tf.zeros(K))

x=tf.placeholder(tf.float32,shape[None,M])
t=tf.placeholder(tf.float32,shape[None,K])
y=tf.nn.softmax(tf.matmul(x,W)+b)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(t*tf.log(y),reduction_indices=[1]))
