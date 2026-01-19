# Smart IoT Energy Platform  
**AI Destekli Akıllı Ev Enerji Analitiği ve Tahmin Sistemi**

##  Proje Özeti
Bu proje, akıllı ev ortamlarında enerji tüketimini izlemek, analiz etmek ve tahmin etmek amacıyla geliştirilmiş uçtan uca bir IoT + Yapay Zeka platformudur.  
Sistem; sensörlerden (gerçek veya dataset tabanlı) gelen enerji verilerini toplayarak:

- Anomali (arıza / olağandışı tüketim) tespiti yapar
- Gelecek enerji tüketimini zaman serisi modelleriyle tahmin eder
- Sistem yük durumu hakkında yorumlayıcı kararlar üretir
- Akademik olarak değerlendirilebilir, açıklanabilir AI çıktıları sunar

Bu çalışma **bitirme projesi** ve **TÜBİTAK başvurusu** kapsamında, gerçek sistem mimarisi gözetilerek tasarlanmıştır.

---

##  Projenin Amacı
- Akıllı evlerde enerji tüketiminin **anlaşılabilir ve öngörülebilir** hale getirilmesi
- Yapay zekânın **sadece grafik değil, ölçülebilir sonuç** üretmesi
- Simülatör bağımlılığı olmadan, **gerçekçi veriyle çalışan** bir altyapı kurulması
- Gerçek donanım (ESP32 vb.) entegrasyonuna hazır bir mimari sunulması

---

##  Sistem Mimarisi (Uçtan Uca Akış)

Veri Kaynağı (Dataset / Sensör)
↓
MQTT (Gerçek Zamanlı Veri Taşıma)
↓
Backend API (.NET)
↓
TimescaleDB / PostgreSQL
↓
AI Servisleri (Python)
├─ Anomali Tespiti
├─ Enerji Tahmini (GRU)
└─ Yorumlayıcı Karar Mekanizması
↓
Sonuç Analizi & Dashboard (HTML)

---

## 📁 Proje Klasör Yapısı ve Görevleri

### `backend/`
.NET tabanlı API servislerini içerir.

- **Amaç:**  
  MQTT üzerinden gelen verileri almak, veritabanına yazmak ve frontend/AI servislerine API sağlamak.
- Gerçek sistemde MQTT + TimescaleDB entegrasyonu bu katmandadır.
- Bu repo sürümünde backend **çalışır yapı** olarak korunmuştur, ancak AI sonuçları offline analiz üzerinden üretilmiştir.

---

### `ai-service/`
Gerçek zamanlı AI servislerini barındırır.

- `ai_anomaly_watcher.py`  
  - MQTT üzerinden gelen verilerde **anlık anomali tespiti** yapar
  - Sistem davranışındaki sapmaları izler
- Simülatör, fake veri ve test script’leri **tamamen kaldırılmıştır**
- Amaç:  
  > “Bu sistem gerçek sensör verisiyle çalışmaya hazırdır” mesajını net vermek

---

### `datasets/`
Çalışmada kullanılan gerçekçi enerji tüketim veri setleri.

- Kaggle kaynaklı, çok cihazlı, zaman serisi yapısında veri
- Simülatör yerine **dataset-driven AI** yaklaşımı benimsenmiştir

---

### `ai-results/`  (Bu projenin akademik kalbi)
Yapay zekâ eğitim, test ve sonuçlarının üretildiği bölüm.

#### İçerik:
- `train_gru_forecast.py`  
  GRU tabanlı zaman serisi model eğitimi
- `train_gru_forecast_optimized.py`  
  Hiperparametre ve pencere uzunluğu (SEQ_LEN) optimizasyonu
- `analyze_ai_results.py`  
  Tüm sonuçları derleyip **tek bir JSON çıktısı** üretir
- `results.json`  
  Dashboard’un beslendiği **nihai akademik sonuç dosyası**
- `dashboard.html`  
  AI çıktılarının yorumlandığı **sunum ekranı**
- `runs_optimized/`  
  - Farklı SEQ_LEN değerleri (24 / 48 / 72)
  - Loss grafikleri
  - Test tahmin grafikleri
  - Model karşılaştırmaları

---

##  Yapay Zekâ Nerede ve Nasıl Kullanıldı?

### 1️⃣ Enerji Tahmini (Forecasting)
- **Model:** Stacked GRU (Gated Recurrent Unit)
- **Neden GRU?**
  - Zaman serilerinde geçmiş bağımlılıklarını yakalar
  - LSTM’ye göre daha hafif ve kararlı
- **Teknik Detaylar:**
  - log1p hedef dönüşümü
  - Lag ve rolling istatistikler
  - Huber loss (outlier dayanımı)
  - Early stopping

**Çıktılar:**
- Test MAPE (%)
- Normalize performans skoru (0–100)
- Eğitim / doğrulama loss grafikleri
- Gerçek vs. tahmin zaman serisi

---

### 2️⃣ Anomali Tespiti
#### a) İstatistiksel Baseline (AI değil, referans)
- Z-score yöntemi
- Açıklanabilir, hızlı, akademik karşılaştırma için

#### b) ML Tabanlı Anomali
- Isolation Forest
- Çok değişkenli davranışsal sapmaları yakalar

**Amaç:**  
> “Sadece eşik aşımı değil, alışkanlık dışı davranışı da yakalayabiliyoruz”

---

### 3️⃣ Yorumlayıcı AI (Decision Support)
- Tahmin edilen saatlik yük
- Referans kapasite ile karşılaştırılır
- Sistem durumu **nitel olarak sınıflandırılır**:
  - Nominal Yük
  - Yüksek Yük

⚠️ Bu ifade **bilinçli olarak** “anlık güç” değil,  
**saatlik yük göstergesi** olarak sunulmuştur.

---

## 📊 Dashboard Ne Gösteriyor?

Dashboard, **grafik süsü değil**, akademik sonuç ekranıdır.

Gösterilenler:
- Dataset boyutu ve agregasyon bilgisi
- Model performansı (MAPE, skor)
- Eğitim süreci (loss eğrileri)
- Test seti tahmin başarısı
- Anomali oranları (baseline vs ML)
- Metodoloji gerekçeleri
- Sınırlılıklar
- Gelecek çalışmalar

---

## ⚠️ Bilinçli Sınırlılıklar
- Tüm ev için saatlik agregasyon yapılmıştır
- Cihaz bazlı forecasting ayrı bir model gerektirir
- İç ortam sensörleri (nem, sıcaklık) eklenirse performans artar

Bu sınırlılıklar **saklanmamış**, özellikle belirtilmiştir.

---

##  Gelecek Çalışmalar
- Cihaz bazlı GRU modelleri (klima, ısıtıcı)
- Autoencoder tabanlı derin anomali tespiti
- MQTT akışında **online inference**
- ESP32 + gerçek sensör entegrasyonu
- Enerji tasarruf öneri motoru

---

##  Akademik Katkı
Bu proje:
- Simülatör bağımlılığını terk etmiş
- Gerçekçi veriyle AI eğitmiş
- Modelin **nasıl öğrendiğini ve nerede sınırlı olduğunu** açıkça göstermiştir

Bitirme projesi ve TÜBİTAK değerlendirmesi için:
- Teknik
- Şeffaf
- Savunulabilir
bir yapı sunar.

---

