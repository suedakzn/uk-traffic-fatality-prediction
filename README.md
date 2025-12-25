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

Bu proje, **Miuul Veri Bilimi Bootcamp** kapsamında ekip çalışması olarak geliştirilmiştir.Birleşik Krallık’ın resmi **STATS19** trafik kazası veri seti kullanılarak, **aşırı dengesiz (imbalanced)** bir problem üzerinde bir  sınıflandırma modeli oluşturulmuştur.

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

