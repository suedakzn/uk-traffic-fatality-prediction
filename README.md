# uk-traffic-fatality-prediction

Bu proje, Birleşik Krallık’ta meydana gelen trafik kazalarına ait veriler kullanılarak **ölümcül (fatal) kazaların önceden tahmin edilip edilemeyeceğini** inceleyen kapsamlı bir **veri bilimi ve makine öğrenmesi** çalışmasıdır.Amaç, yalnızca kaza sayılarını incelemek değil; **hangi koşullarda kazaların ağırlaştığını** ve bu risklerin bir model tarafından ne ölçüde yakalanabildiğini ortaya koymaktır.
---
## 📦 Veri Seti

- **Kaynak:** Kaggle – *Traffic Flow: England, Scotland & Wales (2005–2014)*
- **Orijinal veri aralığı:** 2005 – 2014
- **Bu projede kullanılan dönem:** **2009 – 2014**
- **Filtreleme nedeni:** 
  Veri tutarlılığı sağlamak ve modelleme sürecinde
  eksik/uyumsuz kayıtları minimize etmek.

- **Veri bağlantısı:**  
  https://www.kaggle.com/datasets/daveianhickey/2000-16-traffic-flow-england-scotland-wales

---
## 📖 Proje Hakkında

Bu proje, **Miuul Veri Bilimi Bootcamp** kapsamında ekip çalışması olarak geliştirilmiştir.Birleşik Krallık’ın resmi trafik kazası veri seti kullanılarak, **aşırı dengesiz (imbalanced)** bir problem üzerinde bir  sınıflandırma modeli oluşturulmuştur.

Çalışmanın temel odak noktaları:
- Fatal kazaların kaçırılmaması (**recall odaklı yaklaşım**)
- **Threshold tuning** ile iş problemine uygun karar eşiği seçimi
- **SHAP** kullanılarak model kararlarının yorumlanabilir hale getirilmesi

---
## 🎯 Proje Hedefleri

- Trafik kazalarında **ölümcül risk oluşturan faktörleri** analiz etmek  
- Dengesiz veri üzerinde **etkili bir sınıflandırma modeli** geliştirmek  
- **False Negative (kaçırılan fatal)** vakaları detaylı şekilde incelemek  
- Model çıktılarının **karar destek sistemlerinde** nasıl kullanılabileceğini göstermek

---
## 👥 Proje Ekibi

| İsim | LinkedIn | GitHub |
|:--|:--:|:--:|
| **Süeda Kazan** | [LinkedIn](https://www.linkedin.com/in/sueda-kazan/) | [GitHub](https://github.com/suedakzn) | 
| **Herdem Özen** | [LinkedIn](https://www.linkedin.com/in/herdemozen/) | - | 

---
## 🔍 Veriden Karara: Proje Süreci
### 1️. Keşifsel Veri Analizi (EDA)
- Zamansal analizler: yıl, ay, gün ve saat bazında kaza yoğunlukları  
- Mekânsal analizler: kaza koordinatları kullanılarak **harita tabanlı yoğunluk analizi (Folium)**  
- Kırsal / yerleşim yeri bazlı kaza dağılımları  
- Kaza şiddeti ile;
  - ışık koşulları,
  - hava durumu,
  - yol tipi ve hız limiti
  arasındaki ilişkilerin incelenmesi  

---

### 2️. Feature Engineering
- Tarih (`Date`) ve saat (`Time`) sütunlarının birleştirilmesiyle **`Timestamp`** oluşturulması  
- Zamana dayalı yeni değişkenler:
  - yıl, ay, gün, saat, gün adı, ay adı  
- Kategorik değişkenlerin **One-Hot Encoding** ile modele uygun hale getirilmesi  
- Model performansını etkilemeyen veya bilgi sızıntısına yol açabilecek değişkenlerin elenmesi  

---

### 3️. Modelleme & Karşılaştırma
- **Binary classification:** fatal vs non-fatal  
- Dengesiz veri yapısı için özel yaklaşım:
  - `scale_pos_weight` kullanımı  
- Denenen modeller:
  - **LightGBM Classifier**
  - **XGBoost Classifier**
- Modeller arası karşılaştırma:
  - Recall, Precision, F1-score ve ROC-AUC metrikleri  
- **Stratified Train-Test Split** ile sınıf oranlarının korunması  

---

### 4️. Threshold Tuning & FN (False Negative) Analizi
- Dengesiz sınıf yapısı nedeniyle varsayılan **0.5** eşiği yerine **threshold optimizasyonu** yapıldı.
- Seçilen karar eşiği: **t = 0.20**
- Hedef: **fatal kazaları kaçırmamak (Recall’ı artırmak)** ve Precision–Recall dengesini kontrol etmek.

**FN Analizi (Kaçırılan Fatal Vakalar)**
- Test setindeki toplam fatal sayısı: **2177**
- Modelin kaçırdığı fatal (FN): **200**
- Fatal kaçırma oranı (FN / fatal): **%9.19**

Bu analiz bize şunu sağladı:
- Modelin en çok nerede “emin olamadığını” gördük,
- Özellikle eşik etrafındaki (borderline) vakalarda iyileştirme alanlarını belirledik,
- Threshold/feature geliştirme için aksiyon çıkarabildik.

---

### 5️. Model Yorumlanabilirliği (SHAP)
- **Global SHAP summary plot** ile modelin genel karar yapısının incelenmesi  
- **Feature importance** analizleri  
- En kritik FN örnekleri için:
  - **SHAP waterfall grafikleri**
  - Modelin neden “fatal değil” kararı verdiğinin açıklanması  
- Yorumlanabilirlik çıktılarının **karar destek perspektifiyle** değerlendirilmesi  

---

### 6️. Mekânsal Görselleştirmeler (Folium)
- Kaza yoğunluklarının **interaktif haritalar** üzerinde gösterimi  
- Fatal ve non-fatal kazaların mekânsal karşılaştırması  
- Yüksek riskli bölgelerin görsel olarak öne çıkarılması

---
## 📌 Model Performansı (Threshold Tuning Sonuçları)

LightGBM modeli için farklı threshold değerlerinde **Precision–Recall–F1** karşılaştırıldı.  
Amaç, dengesiz veri yapısında **fatal vakaları mümkün olduğunca kaçırmamak (Recall)** ve kabul edilebilir bir **Precision** seviyesini korumaktı.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.03 | 0.012 | 0.997 | 0.023 |
| 0.05 | 0.012 | 0.994 | 0.024 |
| 0.10 | 0.013 | 0.975 | 0.026 |
| 0.15 | 0.014 | 0.946 | 0.028 |
| 0.20 | 0.016 | 0.908 | 0.031 |

### Seçilen Eşik: **t = 0.20**
- Threshold arttıkça **Recall düşüyor** (daha az fatal yakalanıyor)  
- Ama **Precision ve F1 artıyor** (daha az false alarm, daha dengeli skor)  
- Bu yüzden **Recall’ı hâlâ yüksek tutarken F1’i iyileştiren** bir nokta olarak **0.20** tercih edildi.

---
## 🧰 Kullanılan Teknolojiler & Kütüphaneler

Bu projede analiz, görselleştirme, harita tabanlı keşif ve modelleme adımlarında aşağıdaki araçlar kullanılmıştır:

### 🔹 Veri İşleme & Analiz
- **Python**
- **pandas**, **numpy**  
  → Veri okuma, temizleme, dönüşüm ve feature üretimi

### 🔹 Görselleştirme
- **matplotlib**, **seaborn**
- **missingno**  
  → Eksik değer analizi ve veri kalitesi kontrolleri

### 🔹 Harita Tabanlı Görselleştirme (Folium)
- **folium**
- `HeatMap`, `HeatMapWithTime`, `MarkerCluster`, `FastMarkerCluster`  
  → Kaza yoğunluğu, zamanla değişen yoğunluk ve cluster görselleştirmeleri  
- (Opsiyonel) **DBSCAN (sklearn)**  
  → Yoğun bölgeleri otomatik kümelendirme (hotspot keşfi)

### 🔹 Modelleme (Sadece Boosting)
- **LightGBM (LGBMClassifier)**
- **XGBoost (XGBClassifier)**  
  → İkili sınıflandırma: **fatal vs non-fatal**  
  → Imbalanced problem için `scale_pos_weight` yaklaşımı

### 🔹 Model Değerlendirme & Eşik Optimizasyonu
- **scikit-learn**
  - `train_test_split`, `StratifiedKFold`
  - `classification_report`, `confusion_matrix`
  - `precision_score`, `recall_score`, `f1_score`
  - `ConfusionMatrixDisplay`
  - `cross_validate`  
  → Model performansı ve threshold tuning süreci

### 🔹 Hiperparametre Optimizasyonu
- **GridSearchCV**, **RandomizedSearchCV**  
  → LightGBM / XGBoost için parametre arama

### 🔹 Model Yorumlanabilirliği
- **SHAP**  
  → Global feature importance + örnek bazlı açıklamalar (waterfall)  
  → Özellikle **False Negative (FN)** vakaların analizi

### 🔹 Ek Analiz
- **statsmodels**
  - `variance_inflation_factor (VIF)`  
  → Çoklu doğrusal bağlantı (multicollinearity) kontrolü (opsiyonel)

### 🔹 Yardımcı Araçlar
- `re`, `warnings`, `pathlib`  
  → Feature name cleaning, uyarı bastırma, dosya yolu yönetimi

---
## 📝 Lisans
Bu proje eğitim ve portföy amaçlı geliştirilmiştir. Miuul Data Science Bootcamp kapsamında tamamlanmıştır.
