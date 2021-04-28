Tensorflow Lite Model Maker ile Uygulama için Hazır Edilmesi (Mobil Kullanım için)

pip install tflite-model-maker

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker import model_spec

import matplotlib.pyplot as plt

# Eğitim, Değerlendirme ve Test Verilerinin Yüklenmesi 
training_data = ImageClassifierDataLoader.from_folder('/content/drive/MyDrive/onayli_onaysiz_512x512/training/')
validation_data = ImageClassifierDataLoader.from_folder('/content/drive/MyDrive/onayli_onaysiz_512x512/validation/')
test_data = ImageClassifierDataLoader.from_folder('/content/drive/MyDrive/onayli_onaysiz_512x512/test/')
print(len(validation_data))
print(len(test_data))

# Modelin Eğitilmesi
model = image_classifier.create(training_data, validation_data=validation_data, epochs=10)

# Modelin Test Verileriye Onaylanması
loss, accuracy = model.evaluate(test_data)

# Tensorflow Lite Modelin Yüklenmesi ve bu Dosya Android Studio tarafından kullanılabilecektir
model.export(export_dir='.')
