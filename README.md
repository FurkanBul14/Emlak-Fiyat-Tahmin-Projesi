# 🏠 Ev Fiyat Tahmin Projesi (Housing Price Prediction)

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange) ![Status](https://img.shields.io/badge/Durum-Tamamlandı-green)

Bu proje, makine öğrenmesi tekniklerinden **Basit Doğrusal Regresyon (Simple Linear Regression)** kullanılarak, evlerin metrekare büyüklüğüne göre fiyatını tahmin eden bir yapay zeka modelidir.

---

## 📊 1. Veri Analizi ve Keşif (EDA)

Modeli kurmadan önce veriyi tanımak ve hangi değişkenlerin fiyat üzerinde etkili olduğunu görmek için görselleştirmeler yaptık.

### 🔍 Değişkenler Arası İlişki (Korelasyon)
Veri setindeki sayısal değişkenlerin birbirleriyle olan ilişkisini incelemek için **Isı Haritası (Heatmap)** kullandık.

![Korelasyon Matrisi](ss3.png)
*(Yukarıdaki grafikte görüldüğü üzere, `price` (Fiyat) ile en yüksek ilişkiye sahip olan kutucuk `area` (Alan) kutucuğudur. Kırmızı renk, ilişkinin güçlü olduğunu gösterir.)*

---

### 📈 Alan ve Fiyat Dağılımı
Seçtiğimiz `area` değişkeni ile `price` hedef değişkeninin nasıl dağıldığını görmek için saçılım (scatter) grafiği çizdirdik.

![Dağılım Grafiği](ss1.png)
*(Bu grafik bize evlerin metrekareleri arttıkça fiyatlarının da genel olarak arttığını kanıtlıyor. Noktaların sağ yukarı doğru giden bir trend izlemesi, Doğrusal Regresyon kullanabileceğimizi gösteriyor.)*

---

## 🧹 2. Veri Ön İşleme (Preprocessing)

Ham veri seti üzerinde modelin hatasız çalışması için şu işlemler yapıldı:
* **Eksik Veri Temizliği:** `.dropna()` komutu ile boş (null) değerler temizlendi.
* **Değişken Seçimi:** Analiz sonucunda fiyatı en iyi açıklayan `area` sütunu seçildi. Diğer gürültü oluşturabilecek sütunlar çıkarıldı.
* **Veri Bölme:** Veri seti **%80 Eğitim** ve **%20 Test** olarak ayrıldı.

---

## 🤖 3. Model Sonuçları ve Başarı

Model eğitildikten sonra test verileri üzerinde tahminler yaptı ve gerçek sonuçlarla karşılaştırıldı.

![Regresyon Sonucu](ss2.png)

### 📝 Grafik Yorumu:
* **Mavi Noktalar:** Gerçek ev fiyatlarıdır.
* **Kırmızı Çizgi:** Makinenin öğrendiği "Fiyat Tahmin Doğrusu"dur.
* Çizginin noktaların yoğun olduğu bölgenin tam ortasından geçmesi, modelin genel mantığı başarıyla öğrendiğini gösterir.

### 🏆 Başarı Skoru
Modelin başarısını ölçmek için R2 Skoru kullanılmıştır. Tek bir değişken kullanılmasına rağmen model, fiyat değişimlerini mantıklı bir şekilde açıklayabilmektedir.

---

## 💻 Nasıl Çalıştırılır?

1.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
2.  `Housing.csv` dosyasının proje klasöründe olduğundan emin olun.
3.  Notebook dosyasını çalıştırın.

---

## 👨‍💻 Geliştirici Notu
Bu çalışma, Makine Öğrenmesi dersi kapsamında **veri temizleme, görselleştirme, modelleme ve sonuç yorumlama** süreçlerini uçtan uca uygulamak amacıyla yapılmıştır.
