import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]
# 데이터 상의 관계를 찾아보자.
# 딥러닝 순서 1. 모델만들기 2. 학습

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수(a,b):
    예측_y = train_x * a + b   
    #텐서플로 상에서는 리스트에 곱 합 가능 각각적용해줌
    return tf.keras.losses.mse(train_y, 예측_y)
    #mse : mean squared error

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(3000) :
    opt.minimize( lambda: 손실함수(a,b), var_list=[a,b]) 
    #minimize함수 안에선 단일함수만 들어가야함
    print(a.numpy(),b.numpy())
