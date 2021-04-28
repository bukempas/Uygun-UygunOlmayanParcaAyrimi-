# Uygun/Uygun Olmayan Parca Ayrimi (CNN-TfLite-Object Detection)

## Bilgisayar Görüntüsü (Computer Vision) ve Evrişimsel Sinir Ağları (CNN) kullanarak parça ayrımı yapabilmek

“Esnek ve Hızlı Üretim” noktasına ulaşmanın yolu, kaliteli parça kullanımı ve kaliteli parça/organ üretimlerinin hızlı ayıklanması/onaylanması ile gerçekleşecektir. Mevcut yöntemlerden en çok kullanılanlardan olan elle ve gözle yapılan kontroller artık bu noktada cevap veremeyecektir.
Derin Öğrenme Evrişimsel Sinir Ağları Algoritması (Uygun/Uygun Olmayan sınıflandırması için) ve Computer Vision (Bilgisayar ile Görüntü İşleme) kullanarak parça/organ onayları ve ayıklanması daha hızlı, daha ekonomik ve daha dijital hale getirilebilir.

Örnek olarak aşağıda linki bulunan Kaggle sitesindeki aynı amaçlı bir uygulamanın(yarışmanın) veri setleri kullanılacaktır. Buradaki Veri Arttırımı (Data Augmentation) yapılmış örnek veri setleri My Drive dosyası içine aktarılarak kodlama işlemleri gerçekleşmektedir.
/content/drive/MyDrive/onayli_onaysiz_512x512/casting_data/ dosyası içinde Train ve Test adlı dosyalar altında def_front(uygun olmayan) ve ok_front(uygun) olarak etiketli görüntülerin olduğu dosyalar oluşturulmuştur.
İlgili Kaggle link : https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product


Daha iyi sonuç alabilmek için Öğrenme Aktarımı (Transfer Learning) kullanılmaktadır. Görüntü boyutları 250x250x3 olarak ele alınacaktır. Ayrıca İnce Ayar (Fine Tuning) yapılmayacağı için layer.trainable=False alınarak sadece sınıflandırma katmanlarındaki ağırlıklar eğitilecektir.
 
 " tflearning_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (250,250,3))
   for layer in tflearning_model.layers:
    layer.trainable = False "

### Modelleme sonucu Confusion Matrix ve Classification Report :

![image](https://user-images.githubusercontent.com/59708846/116383467-885a3d80-a81f-11eb-80b4-9d2b429c91cc.png)

              precision    recall  f1-score   support

    Onaysiz     1.0000    0.9911    0.9955       448
     Onayli     0.9847    1.0000    0.9923       257

    accuracy      -         -        0.9943      705
    macro avg    0.9923    0.9955    0.9939      705
    wghtd avg    0.9944    0.9943    0.9943      705


## TensorFlow Lite Model 
Ayrıca TensorFlow Lite Model Maker ile bir model oluşturup bunu Android Studio ile uygulama haline getirebiliriz. Bunun için de modeli bulabilirsiniz.


## Onaysız parçalardaki uygunsuzlukları Nesne Tanımlama ile Tespit Etmek
Bunun için RetinaNet mimarisinden Öğrenme Aktarımı ve İnce Ayar(önceden eğitilmiş COCO checkpoint) 
ile sadece 5 eğitim görüntü veri seti ile yeni görüntüleri hızlı şekilde test edilebilir. Çalışma Zamanı GPU seçilerek hızlı şekilde sonuçlar alınacaktır.
