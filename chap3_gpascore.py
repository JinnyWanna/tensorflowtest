# admit : 합불여부 / gre: 영어성적 / GPA : 학점 / rank : 학교랭킹
import pandas as pd

data = pd.read_csv('tensorflow_course/gpascore.csv')

# print(data.isnull().sum())
# data = data.fillna() # 빈칸을 채워줌

data = data.dropna() # 빈칸제거

x데이터 = []
y데이터 = data['admit'].values


for i, rows in data.iterrows(): # i: index, rows: 해당 행 정보
    #data.iterrows() : dataframe(4개의 열)을 한 행씩 출력할수있다.

    x데이터.append([ rows['gre'], rows['gpa'], rows['rank'] ])

import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'), # 일반적인 히든 레이어
    tf.keras.layers.Dense(128, activation='tanh'), #보통 2의 제곱수로 지정
    tf.keras.layers.Dense(1, activation='sigmoid'), #마지막은 하나의 레이어 (결괏값에 맞는 레이어수 지정) #sigmoid 는 결과를 0~1 사이에서 출력함 (함수모양 참고)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# binary_crossentropy : 확률 예측할 때 이용
# optimizer은 adam 많이 이용

model.fit(np.array(x데이터), np.array(y데이터), epochs= 3000)  
# model.fit(학습데이터 : [ [], [], [], [] ... ], 실제정답 : [ , , , , , ], epochs:몇번 학습할지) // 파이썬 리스트를 넣을순 없음, numpy array 또는 tf tensor로 넣어야함


#예측
예측값 = model.predict( [[750, 3.70, 3],[400, 2.2, 1]] )
print(예측값)

# 모델 생성, 데이터 전처리 , 예측값 출력 순
