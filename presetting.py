import os
import shutil

# print( len( os.listdir('./train/') ) )

for i in os.listdir('./train/'):
  if 'cat' in i:
    shutil.copyfile('./train/' + i, './dataset/cat/' + i)
  if 'dog' in i:
    shutil.copyfile('./train/' + i, './dataset/dog/' + i)

