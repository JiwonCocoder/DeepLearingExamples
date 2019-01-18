import numpy as np
import tensorflow as tf
#１．모델의　파라미터를　정의
#변수를　생성하려면　tf.Varaible()을　호출해야．텐서플로의　독자적인　데이터형으로　데이터를　다룰　수　있다．
#tf.zeros()는　NumPy에　있는　np.zeros()에　해당하는　메서드이며，요소가　０인　다차원　배열을　생성
#웨이트w와　바이어스b를　초기화　한　상태
w= tf.Variable(tf.zeros([2,1]))
b=tf.Variable(tf.zeros([1]))
#print로는　텐서플로의　데이터형에　관한　정보가　출력됨．
print(w)
#2.실제　모델을　구축．
'''
모델의　출력을　나타내는　것은　y=...부분이고
y를　정의하는데　필요한　입력x와　그에　관련된　정답을　나타내는　출력　t를　먼저　정의
텐서플로를　사용하면，　함수를　정의하지　않고，　수식의　모양대로　구현할　수　있어　매우　직관적．
함수로　정의했을　때와　마찬가지로　여기서　y는　실제　값을　포함하지　않으며，x가　함수의　인수에　해당
이런　구조를　가능하게　해주는　것이　바로　x와　t행에　있는　tf.placeholder()
＂placeholder"라는　이름에서　알　수　있듯이　이것은　데이터를　넣어두는　＇그릇＇과　같은　존재이며，
모델을　정의할　때는　데이터의　차원만　정해둔　후，모델을　학습시킬　때처럼
실제로　데이터가　필요한　시점에　값을　넣어　식을　평가할　수　있게　한다．
'''
x=tf.placeholder(tf.float32,shape=[None,2])
'''
None은　데이터　개수를　가변으로　지정할　수　있는　그릇을，
2는　입력　벡터의　차원을
'''
t=tf.placeholder(tf.float32,shape=[None,1])
y=tf.nn.sigmoid(tf.matmul(x,w)+b)
#3.교차　엔트로피　오차　함수：파라미터의　최적화를　위해
cross_entropy=-tf.reduce_sum(t*tf.log(y)+(1-t)*tf.log(1-y))
'''
tf.reduce_sum()은　Numpy의 np.sum()에　해당한다．
'''
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
'''
교차　엔트로피　오차함수의　최적화를　위해　교차　엔트로피　오차　함수를　각　파라미터로　편미분해서　경사를　구하고
（확률）경사하강법을　적용하였다．
GradientDescentOptimizer()에　있는　인수0.1은　학습률을　나타냄．

여기까지가　모델을　학습시키는　부분을　정의하고　프로그램으로　구현한　상황
'''
'''
실제로　학습을　실시하기　전에　학습　후에　나올　결과가　맞는지　확인하는　기능을　구현
로지스틱　회귀에서　모델의　출력 y는　확률이므로，　y>=0.5가　뉴런의　발화　여부를　정하는　기준
'''
correct_prediction=tf.equal(tf.to_float(tf.greater(y,0.5)),t)
#이제부터는　실제롤　학습시키는　부분
#OR 게이트에　대한　학습용　데이터를　정의
X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0],[1],[1],[1]])

#（모델을　정의할　대　선언했던　변수와　식을）초기화
'''
텐서플로에서는　데이터를　취급하는　방법인　세션이라는　것！！
안에서　계산이　실행됨
'''
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#학습
'''
sess.run(train_step)은　경사하강법으로　학습하는　부분에　해당하는데，
여기서 feed_dict를　써서 placeholder인　x와　t에　실제　값을　대입한다．
말그래도　placeholder에　값을　feed(먹이다)하는　것．
에폭　수를　200으로　설정했으므로，　데이터　X를　한번에　모두　넘겨주는　상황．즉　배치　경사하강법을　적용한　것．
'''
for epoch in range(200):
    sess.run(train_step,feed_dict={
        x:X,
        t:Y
    })

#학습결과를　확인
'''
뉴런이　발화하는지　여부를　적절히　분류했는지　확인하려면，　.eval()을　사용해야　한다．
（＋）만약에　tf.Variable()로　정의한　변수라면， sess.run()으로　구할　수　있다．
그런데 correct_prediction에도　실제　값은　들어　있지　않으므로　feed_dict를　써야한다．
이걸　print하면　OR게이트각　제대로　학습됐는지를　확인할　수　있다．
'''
classified=correct_prediction.eval(session=sess,feed_dict={
    x:X,
    t:Y
})
'''
각　입력에　대한　출력　확률을　구하는　부분．
'''
prob=y.eval(session=sess,feed_dict={
    x:X
})
print('w:',sess.run(w))
print('b:',sess.run(b))
print('classified:')
print(classified)
print()
print('output probability:')
print(prob)
