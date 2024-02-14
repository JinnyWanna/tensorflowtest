
# import os
# import tensorflow as tf
# import shutil

# print( len(os.listdir('/content/train/')))

# for i in os.listdir('/content/train/'):
#   if 'cat' in i:
#     shutil.copyfile('/content/train/' + i, '/content/dataset/cat/' + i)
#   if 'dog' in i:
#     shutil.copyfile('/content/train/' + i, '/content/dataset/dog/' + i)


import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/content/dataset/',
  image_size=(64,64), #이미지 사이즈 전처리
  batch_size=64, #배치를 뽑아서, 이미지 64개씩 뽑아서 분석, 반복
  #데이터 ( (xxxx), (yyyy) ) x: 이미지를 숫자로 변환한 것, y : 정답 0 or 1
  subset='training',
  validation_split=0.2, # validation값으로 데이터 20%를 쪼갬
  seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/content/dataset/',
  image_size=(64,64), 
  batch_size=64, 
  subset='validation',
  validation_split=0.2,
  seed=1234
)

# print(train_ds)
# import matplotlib.pyplot as plt


# for i, 정답 in train_ds.take(1):
#   print(i)
#   print(정답)
#   plt.imshow(i[0].numpy().astype('uint8'))
#   plt.show()
# # train_ds가 잘 변환됐는지 보기 위해, plt 라이브러리 이용해서 이미지 출력해보기

#train데이터 80% val데이터 20%

model = tf.keras.models.Sequential([ #input_shape는 trainX가 (60000,28,28)이므로
  tf.keras.layers.Conv2D(32, (3,3) , padding="same", activation="relu", input_shape=(64,64,3)), 
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Conv2D(64, (3,3) , padding="same", activation="relu"), 
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Dropout(0.2), # flatten완화용 레이어, 윗노드들의 레이어를 20% 제거해달라는 뜻 
  #convolution pooling 중에는 보통 pooling끝난 후 Dropout을 진행함.
  tf.keras.layers.Conv2D(128, (3,3) , padding="same", activation="relu"), 
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Flatten(), #평면화하는거 (None, 28, 64)의 데이터를 평면화해서 결과를 단순 10개만 나올수있게 만들어줌
  tf.keras.layers.Dense(128, activation='relu'), 
  #relu: 음수는 다 0으로 만들어주는 활성 함수

  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='sigmoid'),
]) 

model.compile(optimizer="adam",loss="binary_crossentropy", metrics= ['accuracy'])


model.fit(train_ds, validation_data=val_ds epochs= 5)

