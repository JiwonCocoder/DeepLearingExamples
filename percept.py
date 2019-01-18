import numpy as np
'''
여러　개의　데이터를　생성
정규　분포를　따르는　여러　개의　데이터를　생성해야　하는데，
아무렇게나　생성하면　매번　생성되는　데이터　값이　달라지기때문에，　
데이터를　통해　구한　결과도　들쑥날쑥하게　된다．　따라서　계산된　결과에　대한　정확한　평가가　힘들어짐
따라서　매번　＇동일한　랜덤　상태＇를　형성해서　
동일한　조건하에서　결과를　내서　그것을　비교，　평가할　수　있도록　구현해야　함．
'''
rng=np.random.RandomState(123)
d=2
N=10
mean=5

#x1은　발화하지　않는　데이터，　나머지　N은　뉴런이　발화하는　데이터 x2
x1=rng.randn(N,d)+np.array([0,0])
x2=rng.randn(N,d)+np.array([mean,mean])

#생성한　두　종류의　데이터를　쉽게　처리할　수　있게　x1,x2를　합쳐둔　것
x=np.concatenate((x1,x2),axis=0)
#모델에　필요한　파라미터인　w와　b를　초기화
w=np.zeros(d)
b=0

#출력식을　프로그램　함수로　정의하면
def y(x):
    return step(np.dot(w,x)+b)

def step(x):
    return 1*(x>0)

#파라미터를　계속　갱신하려면　제대로　된（정답인）　출력값이　필요하므로　이　출력값을　다음과　같이　정의합니다

def t(i):
    if i<N:
        return 0
    else:
         return 1

#d오차정정학습법에서는　모든　데이터가　제대로　분류될　때까지　학습을　반복．
'''반복부분'''
while True:
    #파라미터　갱신　처리
    classified=True
    for i in range(N*2):
        delta_w=(t(i)-y(x[i]))*x[i]
        delta_b=(t(i)-y(x[i]))
        w+=delta_w
        b+=delta_b
        #20개중　하나라도　delta_w와 delta_b가　０이　아니면，　
        classified*= all(delta_w==0)*(delta_b==0)
    #모든　데이터가　제대로　분류됐다면＝20개의　데이터　모두의　w,b에　변화가　없는　상황
    if classified:
        break


print(y([0,0]))
print(y([5,5])