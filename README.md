# Smart IoT Platform – AI Destekli Enerji Analizi

Bu proje, akıllı ev ortamlarında oluşan enerji tüketim verilerinin
yapay zeka destekli analizini ve tahminini amaçlamaktadır.

## Proje Amacı
Amaç, enerji tüketim davranışlarını analiz etmek, geleceğe yönelik
tahminler üretmek ve olağandışı (anomalik) tüketimleri tespit ederek
karar destek mekanizması oluşturmaktır.

## Kullanılan Yaklaşım
- Zaman serisi analizi
- GRU (Gated Recurrent Unit) tabanlı derin öğrenme modeli
- İstatistiksel ve makine öğrenmesi tabanlı anomali tespiti
- Sonuçların görsel dashboard ile sunulması

## Veri Kullanımı
Projenin final sürümünde, başlangıç aşamasında test amaçlı kullanılan
simülatör tabanlı veri üretimi tamamen kaldırılmıştır.

Tüm yapay zeka analizleri, gerçekçi bir enerji tüketim veri seti
üzerinden offline olarak gerçekleştirilmiştir.

## AI Modülleri
- **Enerji Tüketim Tahmini:**  
  GRU modeli kullanılarak saatlik enerji tüketimi tahmin edilmiştir.
  Model performansı MAPE metriği ile değerlendirilmiştir.

- **Anomali Tespiti:**  
  - İstatistiksel yöntem (Z-Score)  
  - Makine öğrenmesi tabanlı yöntem (Isolation Forest)

## Sonuçlar
Model eğitim ve test çıktıları, tahmin grafikleri ve performans metrikleri
`ai-results/` klasörü altında yer almaktadır.
Bu sonuçlar, etkileşimli bir HTML dashboard üzerinden incelenebilmektedir.

## Gelecek Çalışmalar
Bu altyapı, gerçek zamanlı sensör verileri (ör. ESP32 tabanlı IoT cihazları)
ile entegre edilerek çevrim içi (online) tahmin ve anomali tespiti
yapabilecek şekilde genişletilebilir.
