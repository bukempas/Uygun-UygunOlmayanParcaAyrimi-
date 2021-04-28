# Uygun-UygunOlmayan-ParcaAyrimi

Bilgisayar Görüntüsü (Computer Vision) ve Evrişimsel Sinir Ağları (CNN) kullanarak parça ayrımı yapabilmek

“Esnek ve Hızlı Üretim” noktasına ulaşmanın yolu, kaliteli parça kullanımı ve kaliteli parça/organ üretimlerinin hızlı ayıklanması/onaylanması ile gerçekleşecektir. Mevcut yöntemlerden en çok kullanılanlardan olan elle ve gözle yapılan kontroller artık bu noktada cevap veremeyecektir.
Derin Öğrenme Evrişimsel Sinir Ağları Algoritması (Uygun/Uygun Olmayan sınıflandırması için) ve Computer Vision (Bilgisayar ile Görüntü İşleme) kullanarak parça/organ onayları ve ayıklanması daha hızlı, daha ekonomik ve daha dijital hale getirilebilir.

Örnek olarak aşağıda linki bulunan Kaggle sitesindeki aynı amaçlı bir uygulamanın(yarışmanın) veri setleri kullanılacaktır. Buradaki Veri Arttırımı (Data Augmentation) yapılmış örnek veri setleri My Drive dosyası içine aktarılarak kodlama işlemleri gerçekleşmektedir.
/content/drive/MyDrive/onayli_onaysiz_512x512/casting_data/ dosyası içinde Train ve Test adlı dosyalar altında def_front(uygun olmayan) ve ok_front(uygun) olarak etiketli görüntülerin olduğu dosyalar oluşturulmuştur.
İlgili Kaggle link : https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product


Daha iyi sonuç alabilmek için Öğrenme Aktarımı (Transfer Learning) kullanılmaktadır. Görüntü boyutları 250x250x3 olarak ele alınacaktır. Ayrıca İnce Ayar (Fine Tuning) yapılmayacağı için layer.trainable=False alınarak sadece sınıflandırma katmanlarındaki ağırlıklar eğitilecektir.
 
 " tflearning_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (250,250,3))
   for layer in tflearning_model.layers:
    layer.trainable = False "

Ayrıca kendi veri setimiz olduğu için TensorFlow Lite Model Maker ile bir model oluşturup bunu Android Studio ile uygulama haline getirebiliriz. Bunun için de modeli bulabilirsiniz.


Onaysız parçalardaki uygunsuzlukları Nesne Tanımlama ile tespit etmek edebiliriz.
Bunun için RetinaNet mimarisinden Öğrenme Aktarımı ve İnce Ayar(önceden eğitilmiş COCO checkpoint) 
ile sadece 5 eğitim görüntü veri seti ile yeni görüntüleri hızlı şekilde test edilebilir. Çalışma Zamanı GPU seçilerek hızlı şekilde sonuçlar alınacaktır.
