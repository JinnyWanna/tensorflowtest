import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  './dataset/',
  image_size=(64,64), #이미지 사이즈 전처리
  batch_size=64, #배치를 뽑아서, 이미지 64개씩 뽑아서 분석, 반복
  #데이터 ( (xxxx), (yyyy) ) x: 이미지를 숫자로 변환한 것, y : 정답 0 or 1
  subset='training',
  validation_split=0.2, # validation값으로 데이터 20%를 쪼갬
  seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  './dataset/',
  image_size=(64,64), 
  batch_size=64, 
  subset='validation',
  validation_split=0.2,
  seed=1234
)

def 전처리함수(i, 정답) :
  i = tf.cast( i/255.0, tf.float32 )
  return i, 정답


train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)


model = tf.keras.models.Sequential([ #input_shape는 trainX가 (60000,28,28)이므로
  tf.keras.layers.Conv2D(32, (3,3) , padding="same", activation="relu", input_shape=(64,64,3)), 
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Conv2D(64, (3,3) , padding="same", activation="relu"), 
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(128, (3,3) , padding="same", activation="relu"), 
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Flatten(), #평면화하는거 (None, 28, 64)의 데이터를 평면화해서 결과를 단순 10개만 나올수있게 만들어줌
  tf.keras.layers.Dense(128, activation='relu'), 
  #relu: 음수는 다 0으로 만들어주는 활성 함수
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid'),
]) 

model.compile(optimizer="adam",loss="binary_crossentropy", metrics= ['accuracy'])


model.fit(train_ds, validation_data=val_ds, epochs= 5)