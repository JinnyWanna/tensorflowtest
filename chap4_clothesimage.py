import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#데이터 로드
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()



trainX = trainX / 255.0 
testX = testX / 255.0  
#데이터를 0~255에서 0~1로 압축

trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1)) # Conv2D를 위한 데이터전처리 60000 28 28 1로 변경
testX = testX.reshape( (testX.shape[0],28,28,1))

# ( (trainX, trainY ), (textX, testY) ) 자료형
 
class_names = ['T-shirt/top', 'Trouser', 'Pullove r', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

model = tf.keras.models.Sequential([ #input_shape는 trainX가 (60000,28,28)이므로
    tf.keras.layers.Conv2D(32, (3,3) , padding="same", activation="relu", input_shape=(28,28,1)), 
    tf.keras.layers.MaxPooling2D( (2,2) ),

    # tf.keras.layers.Conv2D(32, (3,3) , padding="same", activation="relu", input_shape=(28,28,1)), 
    # tf.keras.layers.MaxPooling2D( (2,2) ),
    # 컬러인 경우 input_shape마지막은 3
    # 32개의 다른 feature을 생성해라, kernel 가로세로 사이즈(보통 (3,3))
    # padding: 작아진 사이즈를 맞추기위해 padding에 빈 픽셀들을 넣어준다
    # 보통 relu 사용 : 이미지를 숫자로 바꾸면 0~255 사이이므로
    #input_shape는 하나의 자료를 넣으면 되는데, 현재 하나의 자료가 (28,28)이므로 (28,28,1)로 변환해 넣어준다.
    # 컨볼루션, 풀링작업 여러번 가능
    tf.keras.layers.Flatten(), #평면화하는거 (None, 28, 64)의 데이터를 평면화해서 결과를 단순 10개만 나올수있게 만들어줌
    tf.keras.layers.Dense(128, activation='relu'), 
    #relu: 음수는 다 0으로 만들어주는 활성 함수

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
]) 
# softmax도 0~1로 바꿔줌, category예측 문제에 사용 예측한 10개 확률을 모두 더하면 1나옴,
# sigmoid는 binary prediction문제에 사용


# model.summary() #input_shape=()가 있어야 볼수있음

# exit()

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics= ['accuracy'])


model.fit(trainX, trainY, validation_data=(testX, testY), epochs= 5)
#validation_date : epochs끝날때마다 평가

# score = model.evaluate( testX, testY ) #평가
# print(score)
# 위 항목 이용보다는 fit에 validation_data를 넣어줘서 epochs마다 결과 분석 가능.
# predic = model.predict(testX)

#overfitting 현상 : training의 데이터 셋을 기계가 외워서 결괏값이 더 잘나오게됨.
