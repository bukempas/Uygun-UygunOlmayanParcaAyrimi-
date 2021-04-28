Evrişimsel Sinir Ağları(CNN) ve Öğrenme Aktarımı (Tranfer Learning) kullanarak Onaylı/Onaysız Parça Ayrımı


# 1.Kütüphanelerin İndirilmesi

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os,shutil
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Dense, Activation, Flatten, MaxPooling2D
from keras import optimizers

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# 2.My Drive Üzerinden Görüntü Veri Setlerinin Alınması

os.getcwd()

# Çalışma Alanı Değişimi 
# Belirli adrese yönelik 
os.chdir('/content/drive/MyDrive/onayli_onaysiz_512x512/casting_data/') 
  
# getcwd() kullanarak doğrulamak
cwd = os.getcwd() 
  
# Mevcut Çalışma Alanı Bilgisi
print("Çalışma Alanı:", cwd)

print(os.listdir())

# 3.Transfer Learning (InceptionResNetV2 kullanarak)

tflearning_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (250,250,3))
for layer in tflearning_model.layers:
  layer.trainable = False



```

```

# 4.Veri Seti : Eğitim, Onay ve Test olmak üzere 3 ayrı set olarak ele alınıyor.

target_size=(250,250)
train_datagen = ImageDataGenerator(validation_split=0.2,
    rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'train',
        subset='training',
        target_size=target_size,
        batch_size=128,
        class_mode='binary')
validation_generator = train_datagen.flow_from_directory(
        'train',
        subset='validation',
        target_size=target_size,
        batch_size=128,
        class_mode='binary')

test_data = ImageDataGenerator( 
    rescale=1./255)

test_generator = test_data.flow_from_directory(
        'test',
        target_size=target_size,
        batch_size=64,
        class_mode='binary',
        shuffle=False)


train_generator.class_indices

# 5.Öğrenme Aktarımı ile Elde Edilen Katmanların, Sınıflandırma Katmanları ile Birleştirilmesi

x = layers.Flatten()(tflearning_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation = 'sigmoid')(x)

from tensorflow.keras import Model
model = Model( tflearning_model.input, x) 

# 6.Compile ve Training Aşaması

from tensorflow.keras.optimizers import RMSprop, Adam, SGD
model.compile(loss='binary_crossentropy',
              optimizer = 'Adam',
              metrics=['accuracy'])
model.summary()

EPOCHS=10
history = model.fit(
      train_generator,
      epochs=EPOCHS,
      validation_data=validation_generator,
      verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()

# 7.Yeni Veriler ile Modelin Test Edilmesi

import seaborn as sns
from sklearn import metrics

# Test dosyasındaki görüntülerle tahminleme yapmak (olasılık sonucu 0,5 altı uygun olmayan, 0,5 ve üstü uygun)
pred_proba = model.predict(test_generator, verbose=1)
threshold = 0.5
pred = pred_proba >= threshold

# Karmaşıklık Matrisi (confusion matrix)
plt.figure(figsize=(6,4))
sns.heatmap(
    metrics.confusion_matrix(test_generator.classes,pred),
    annot=True,
    annot_kws={'size':14, 'weight':'bold'},
    fmt='d',
    xticklabels=['onaysiz', 'onayli'],
    yticklabels=['onaysiz', 'onayli'])
plt.tick_params(axis='both', labelsize=14)
plt.ylabel('Actual', size=14, weight='bold')
plt.xlabel('Predicted', size=14, weight='bold')
plt.show()

target_names = ['Onaysiz', 'Onayli']
print(metrics.classification_report(
    test_generator.classes, pred, target_names=target_names, digits = 4))

# 8.Modelin Kaydedilmesi ve Yeniden Kullanılması(Yuklenmesi)

model.save('def_ok_InceptionResnetv2.h5')

from keras.models import load_model
model = load_model('/content/drive/MyDrive/onayli_onaysiz_512x512/def_ok_InceptionResnetv2.h5')

