# Uygun-UygunOlmayan-ParcaAyrimi

Bilgisayar Görüntüsü (Computer Vision) ve Evrişimsel Sinir Ağları (CNN) kullanarak parça ayrımı yapabilmek

“Esnek ve Hızlı Üretim” noktasına ulaşmanın yolu, kaliteli parça kullanımı ve kaliteli parça/organ üretimlerinin hızlı ayıklanması/onaylanması ile gerçekleşecektir. Mevcut yöntemlerden en çok kullanılanlardan olan elle ve gözle yapılan kontroller artık bu noktada cevap veremeyecektir.
Derin Öğrenme Evrişimsel Sinir Ağları Algoritması (Uygun/Uygun Olmayan sınıflandırması için) ve Computer Vision (Bilgisayar ile Görüntü İşleme) kullanarak parça/organ onayları ve ayıklanması daha hızlı, daha ekonomik ve daha dijital hale getirilebilir.

Örnek olarak aşağıda linki bulunan Kaggle sitesindeki aynı amaçlı bir uygulamanın(yarışmanın) veri setleri kullanılacaktır. Buradaki Veri Arttırımı (Data Augmentation) yapılmış örnek veri setleri My Drive dosyası içine aktarılarak kodlama işlemleri gerçekleşmektedir.
İlgili link : https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product


Daha iyi sonuç alabilmek için Öğrenme Aktarımı (Transfer Learning) kullanılmaktadır. Görüntü boyutları 250x250x3 olarak ele alınacaktır. Ayrıca İnce Ayar (Fine Tuning) yapılmayacağı için layer.trainable=False alınarak sadece sınıflandırma katmanlarındaki ağırlıklar eğitilecektir.
 
 " tflearning_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (250,250,3))
   for layer in tflearning_model.layers:
    layer.trainable = False "
