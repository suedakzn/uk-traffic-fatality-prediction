""""
############ Veri Setine Genel Bakış ###################################################################################

#Bu veri seti, 2009 ile 2014 yılları arasında Birleşik Krallık’ta meydana gelen trafik kazaları
# hakkında ayrıntılı bilgiler içermektedir.
#Veriler iki ayrı CSV dosyasına bölünmüştür ve bir milyona yakin deger  kapsamaktadır.
#Kazaların ciddiyeti, hava durumu, ışık koşulları, yol tipi ve olay yerine bir polis memurunun
# gelip gelmediği gibi çok çeşitli faktörleri içermektedir.


#####   Sütun_Isimleri                         Tanimlari (TR) ##########################################################
#Accident_Index                                 Her kaza için index bilgisi
#Location_Easting_OSGR                          Dogu koordinatı (British National Grid)
#Location_Northing_OSGR                         Kuzey koordinatı (British National Grid)
#Longitude                                      Coğrafi boylam
#Latitude                                       Coğrafi enlem
#Police_Force                                   Sorumlu polis birimi
#Accident_Severity                              Kaza şiddeti (1: Ölümcül, 2: Ciddi, 3: Hafif)
#Number_of_Vehicles                             Kazaya karışan araç sayısı
#Number_of_Casualties                           Yaralı veya ölü sayısı
#Date                                           Kazanın gerçekleştiği tarih
#Day_of_Week                                    Kazanın gerçekleştiği gün
#Time                                           Kaza zamanı
#Local_Authority_(District)                     Kazanın meydana geldiği yerel idare bölgesi
#Local_Authority_(Highway)                      Sorumlu karayolu idaresi
#1st_Road_Class                                 Birinci yolun sınıflandırması
#1st_Road_Number                                Birinci yolun numarası
#Road_Type                                      Yol türü (ör. tek yön, çift şeritli)
#Speed_limit                                    Kaza yerindeki hız sınırı
#Junction_Detail                                Kavşak tipi
#Junction_Control                               Kavşak kontrol türü (ör. trafik ışığı)
#2nd_Road_Class                                 İkinci yolun sınıflandırması (uygunsa)
#2nd_Road_Number                                İkinci yolun numarası
#Pedestrian_Crossing-Human_Control              İnsan kontrollü yaya geçidi bulunup bulunmadığı
#Pedestrian_Crossing-Physical_Facilities        Fiziksel yaya geçidi imkânları
#Light_Conditions                               Olay yerindeki ışık koşulları (ör. gündüz, karanlık)
#Weather_Conditions                             Kaza sırasındaki hava koşulları
#Road_Surface_Conditions                        Yol yüzeyi koşulları (ör. kuru, ıslak)
#Special_Conditions_at_Site                     Olay yerinde bildirilen özel koşullar
#Carriageway_Hazards                            Yoldaki tehlikeler (ör. enkaz)
#Urban_or_Rural_Area                            Kentsel için 1, kırsal için 2
#Did_Police_Officer_Attend_Scene_of_Accident    Polis memurunun olay yerine gelip gelmediği
#LSOA_of_Accident_Location                      LSOA (küçük alan coğrafi kodu)
#Year                                           Kazanın gerçekleştiği yıl




##### Column Name	                            Description (EN) #######################################################
# Accident_Index	                             Unique identifier for each accident
# Location_Easting_OSGR	                         Easting coordinate (British National Grid)
# Location_Northing_OSGR                         Northing coordinate (British National Grid)
# Longitude	                                     Geographic longitude
# Latitude	                                     Geographic latitude
# Police_Force	                                 Police department responsible
# Accident_Severity	                             Severity of the accident (1: Fatal, 2: Serious, 3: Slight)
# Number_of_Vehicles	                         Number of vehicles involved
# Number_of_Casualties	                         Number of casualties (injuries or deaths)
# Date	                                         Date of the accident
# Day_of_Week	                                 Day when the accident occurred
# Time	                                         Time of the accident
# Local_Authority_(District)	                 Local government district of the accident
# Local_Authority_(Highway)	                     Highway authority responsible
# 1st_Road_Class	                             Classification of the first road
# 1st_Road_Number	                             Road number
# Road_Type	                                     Type of road (e.g., one way, dual carriageway)
# Speed_limit	                                 Speed limit at the accident location
# Junction_Detail	                             Type of junction where accident occurred
# Junction_Control	                             Junction control type (e.g., traffic signal)
# 2nd_Road_Class	                             Classification of the second road (if applicable)
# 2nd_Road_Number	           	                 Second road number
# Pedestrian_Crossing-Human_Control	             Whether there was a human-controlled crossing
# Pedestrian_Crossing-Physical_Facilities	     Physical pedestrian facilities at the site
# Light_Conditions	                             Lighting at the scene (e.g., daylight, darkness)
# Weather_Conditions	                         Weather conditions during the accident
# Road_Surface_Conditions	                     Road surface conditions (e.g., dry, wet)
# Special_Conditions_at_Site	                 Any special conditions reported at the scene
# Carriageway_Hazards	                         Hazards present on the road (e.g., debris)
# Urban_or_Rural_Area	                         1 for urban, 2 for rural
# Did_Police_Officer_Attend_Scene_of_Accident	 Whether police attended the scene
# LSOA_of_Accident_Location	                     Lower Layer Super Output Area (small area geocode)
# Year	                                         Year of the accident


"""



######    Importing Libraries   ################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
import folium
from folium.plugins import FastMarkerCluster, HeatMap
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from folium.plugins import HeatMapWithTime
from sklearn.cluster import DBSCAN
from pathlib import Path
import webbrowser
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
#
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import re
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate

####### Cikti Ayarlari #################################################################################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



####### Reading and Merging Datasets ###################################################################################################


df_0911 = pd.read_csv("Datasets/accidents_2009_to_2011.csv")
df_1214 = pd.read_csv("Datasets/accidents_2012_to_2014.csv")
df_first= pd.concat([ df_0911, df_1214], ignore_index=True)

df_first.shape
#(934139 gözlem birimi, 33 degisken)

df=df_first.copy()


########################################################################################################################
#                          KEŞİFCİ VERİ ANALİZİ(Exploratory Data Analysis )

########################################################################################################################


def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    num_cols = dataframe.select_dtypes(include=np.number)
    if num_cols.shape[1] == 0:
        print("No numeric columns")
    else:
        print(num_cols.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

df.describe().T

df.describe(include="object").T


df.columns
"""
# Accident_index(object)
 -> her kazaya verilen benzersiz ID
 -> modelde feature olarak kullanilmayacak

# Locaiton_Easting_OSGR, Locaiton_Northing_OSGR(int 64)
 -> İngilteredeki koordinat sisteminde ki x-y koordinatlari
 -> Easting : x (dogu-bati)
 -> Norhing : y (kuzey-guney)
 -> Hot spot analizinde kullanilabilir 
 -> Modelleme yaparken cluster ile bolge cikarmak icin kullanilabilir

# Longitude, Latitude(float 64)
 -> Aynı yerin enlem-boylam bilgisi
 -> Gorsellestirme, mesafe hesaplamada kullabilirim
 -> Bolge/sehir gibi kumeleyip turetip ozellik olarak kullanabilirim

# Police_Force (int 64)
 -> Kazaya bakan polis biriminin kodu
 -> Kategorik degisken gibi gozukuyor
 -> Bazi bolgelerde daha agir kazalar oluyorsa pattern yakalamama yardimci olur

# Accident_Severity (int 64)
 -> Kaza siddeti
 -> 1: Olumcul, 2: Ciddi, 3: Hafif
 -> Hedef degiskenim
 -> Not: İlk bakışta kazaların çoğu 3 (hafif) gibi duruyor, sınıf dengesizliği olabilir.

# Number_of_Vehicles (int 64)
 -> Kayıtlı kazaya karisan arac sayisi
 -> Kazanin boyutu ve karmasikligi hakkinda bilgi verebilir
 -> Aykiri deger riski (67 aracli asiri buyuk zincirleme kaza verisi olabilir) 

# Number_of_Casualties (int 64)
 -> Kaza sonucu yaralanan/olen sayisi toplami
 -> Buradaki deger bana kazanin ciddiyet durumuyla ilgili bilgi verebilir (hedef degiskenle direkt baglantili olabilir!!!)

# Date (object)
 -> Kaza tarihi string sekilde ("x/x/x")
 -> datetime cevrilip yil,ay,gun,mevsim,haftaici,haftasonu gibi featurelar uretmem gerekiyor
 -> Trend analizi yapacagim zaman onemli (yillara gore degisim)

# Day_of_Week (int 64)
 -> 1-7 arasinda kodlanmis haftanin gunleri
 -> Turkiye sisteminden farkli (1: pazar)
 -> Hafta ici ve hafta sonu arasinda risk farkı var mi?
 -> Hangi gunlerde daha agir kazalar oluyor?
 -> Hangi gunlerde oranlar daha dusuk?

# Time (object)
 -> Kazanin saati
 -> Gunu parcalara bolerek feature uretecegim (sabah,ogle,aksam,gece, rush hour)
 -> Eksik değer: 50 adet ( yaklasik %0.01) → oran çok düşük, ister satır drop ederim, ister uygun bir kategoriye (örn. "Unknown") atarım.

# Local_Authority_(District) (int64)
 -> Kaza yerinin bagli oldugu ilce/yerel yonetim kodu
 -> Kategorik ulke kodu (yani int gibi gozukuyor ama aslinda kategorik)
 -> Bolgesel risk analizinde kullanacagim
 -> Hangi bolgede daha cok agir kaza var??

# Local_Authority_(Highway) (int64)
 -> Yolu yoneten/isleten otoritenin kodu 
 -> Sayisal gibi gozuken ama aslinda kategorik olan bir degisken
 -> Farkli yol tasarimlari olabilir 

# 1st_Road_Class (int 64)
 -> Kazanin oldugu birincil yol sinif (otoban, a-road, b-road,...)
 -> Yol sinifina gore kaza siddeti hakkinda ya da sikligi hakkinda bilgi edinebilirim
 -> Kategorik olarak mi yaklasmaliyim ??

# 1st_Road_Number (int 64)
 -> Yolun numarasi ( A1, B432,...)
 -> Tek basina anlamli degil fakat spesifik yollari incelemek istersek kullanabilirim
 -> ID gibi direkt modellemede kullanmamam gerekiyor

# Road_Type (object)
 -> Yolun tipi (tek yönlü, çift yönlü vb)
 -> Yol tipine gore kaza turu ve siddeti ne?
 -> Guzel featureler uretebilirim

# Speed_Limit (int 64)
 -> Kaza anindaki yolun hiz limiti
 -> Hedef degiskenimle yuksek iliskisi olacagini dusunuyorum
 -> Sayisal mi kategorik mi ele alinmali?

# Junction_Detail (float 64)
 -> Kavsak tipi detayi
 -> Eksik değer: 934.139 adet (%100) → bütün satırlar NaN.
 -> Bu yüzden büyük ihtimalle veri setinden tamamen drop edilecek

# Junction_Control (object)
 -> Kavsaktaki kontrol türü (kontrolsüz, otomatik trafik sinyali)
 -> Kavsak kontrolü, kaza riskini ve siddetini etkileyecek bir degisken
 -> Eksik değer: 365.890 adet (~%39).
 -> Eksik oranı yüksek → ya "Unknown" gibi ayrı bir kategori açılabilir ya da değişkenin tamamını kullanmaktan vazgeçilebilir; karar EDA sonrası verilecek.


# 2nd_Road_Class (int 64)
 -> Eger kavsak varsa ikinci yolun sinifi 
 -> -1: ikincil yol yok / bilinmiyor
 -> Sayisal gozuken ama kategorik olan degisken
 -> -1 degerini "No_second_road" seklinde kategorik degiskene donusturebilirim

# Pedestrian_Crossing-Human_Control (object)
 -> Yaya gecidinde insan kontrolü var mi?
 -> Okul gorevlisi, trafik polisi vb
 -> Yaya kazalari icin onemli feature
 -> Kategorik

# Pedestrian_Crossing-Physical_Facilities (object)
 -> Fiziksel yaya gecidi var mi?
 -> Yaya kazalari icin onemli
 -> Kategorik

# Light_Conditions (object)
 -> İsik durumu
 -> Gece/gündüz + aydinlatmanin olup olmamasi durumunda kaza siddeti ne?

# Weather_Conditions (object)
 -> Hava durumunun  hedef degiskene etkisi ne?
 -> Feature olarak iyi
 -> Eksik değer: 106 adet (~%0.01) → oran çok düşük, en sık kategori ile doldurulabilir?

# Road_Surface_Conditions (object)
 -> Yol yüzeyinin durumu
 -> Kayma ya da kontrol kaybi gibi bilgileri cikarabilirim
 -> Feature olarak iyi
 -> Eksik değer: 1.296 adet (~%0.14) → düşük oran, en sık kategori ile doldurma??

# Special_Conditions_at_Site (object)
 -> Yolda ekstra ozel durumlar var mi? (yol calismasi, daralma, gecici trafik duzeni vb)
 -> Eksik değer: 912.232 adet (~%97.7).
 -> Neredeyse tamamen boş olduğu için büyük ihtimalle drop edilecek ya da ??

# Carriageway_Hazards (object)
 -> Yolda tehlike durumu (yag dokulmesi, dokulmus yuk, yol uzerinde nesne vb)
 -> Eksik değer: 918.019 adet (~%98.3).
 -> Yine neredeyse tamamen boş; büyük ihtimalle veri setinden çıkartılacak.

# Urban_or_Rural_Area (int 64)
 -> Sehir ici mi kirsal mi (1:Sehir ici , 2:Kirsal)
 -> Alanin ne oldugu trafik yogunlugu ve hiz siniri gibi bilgileri verecegi icin kaza turu bakimindan kritik degisken

# Did_Police_Officer_Attend_Scene_of_Accident (object)
 -> Olay yerine polis gitmis mi?
 -> Daha ciddi kazalarda polisin gitme durumu olabilir?
 -> Eksik değer: 547 adet (~%0.06) → düşük oran, "Unknown" kategorisi ya da en sık kategori ile doldurulabilir

# LSOA_of_Accident_Location (object)
 -> LSOA: Small area code – küçük istatistiki bölge kodu
 -> Sosyo-ekonomik ve mekansal analiz icin kullanilabilir ama tek basina granular?
 -> Fazla kategori iceriyor direkt modele sokmak sikintili
 -> Eksik değer: 60.727 adet (~%6.5) → orta seviye; kullanıp kullanmamaya EDA sonrası karar verilecek

# Year (int 64)
 -> Kaza yili hakkinda bilgi
 -> Zaman trendi gormek icin kullanicam


#### SAYISAL DEGISKEN DAGILIMI HAKKINDA ####
- Location_Easting_OSGR & Location_Northing_OSGR 

- 
"""


#################################
# DATE-TIME FEATURE CREATION
#################################
"""
Date ve Time sutunlarini birlestirerek Timestamp olusturuldı ve buradan yil,ay,gun ve saat gibi zaman tabanlı
featurelar uretildi. Donusum sonrasinda 50 kayıtta Timestamp uretilmedi (NaT) ve bozuk/eksik satirlar oldugunu
analiz edildi. Bu hatali satirlar toplam verinin cok kucuk bir kismi oldugu icin veri setinden cikarildi.
"""


before_rows = len(df) # 934139
df["Timestamp"] = pd.to_datetime(
    df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
    format="%d/%m/%Y %H:%M",
    errors="coerce")

# Sutun uretme
df["year"] = df["Timestamp"].dt.year
df["month"] = df["Timestamp"].dt.month
df["day_name"] = df["Timestamp"].dt.day_name()
df["month_name"] = df["Timestamp"].dt.month_name()
df["Hour"] = df["Timestamp"].dt.hour

# NaT sayisi ve yuzdesi
nat_count = df["Timestamp"].isna().sum()
nat_rate = (nat_count / before_rows) * 100

print("Timestamp NaT sayısı:", nat_count)
print(f"Timestamp NaT oranı: %{nat_rate:.4f}") # %0.0054


print(df[["Date", "Time", "Timestamp", "year", "month", "day_name", "month_name", "Hour"]].head())

# dropna oncesi-sonrasi karsilastirma
rows_before_drop = len(df)
df.dropna(subset=["Timestamp", "year", "month", "day_name", "month_name", "Hour"], inplace=True)
rows_after_drop = len(df)

dropped = rows_before_drop - rows_after_drop
dropped_rate = (dropped / rows_before_drop) * 100

print("dropna sonrası satır sayısı:", rows_after_drop) # 934089
print("Silinen satır sayısı:", dropped) # 50
print(f"Silinen satır oranı: %{dropped_rate:.4f}") # %0.0054


################################################
# EXPLORARY DATA ANALYSIS
################################################
#----------------- KONUMSALLIK HAKKINDA ANALİZLER -------------------
###### Accidents Density Graph(Where do most accidents occur?)#####
plt.figure(figsize=(5.5,9.5))
plt.axes().set_facecolor("black")
plt.scatter(x = df["Longitude"], y = df["Latitude"],s=0.005, alpha= 0.25, color="lightyellow")
plt.title("UK Accidents 2009-2014")
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig("Accidents Density Graph.png")
plt.show()
# kazaların çoğu, beklendiği gibi, Londra, Liverpool ve Midlands çevresindeki büyük şehirlerde
# İngiltere’de meydana geldiği görülüyor. Bununla birlikte, en çok
# yolculuğun da burada yapıldığını unutmamak gerek. Muhtemelen İngiliz sürücüler nasıl araç kullanacaklarını bilmedikleri
# için değil. Trafik yoğun olduğunda kaza yaşama olasılığı daha yüksek olduğu için bu sonuç ortaya çıkıyor. Aynı mantık
# tüm analiz boyunca uygulanabilir.”


#----------------- ZAMANSAL KOŞULLAR HAKKINDA ANALİZLER -------------------
#***** Mevsimlere Göre Kaza Oranı *****
# Burada mevsim bazında kazaların toplam içindeki payını (%) görmek istedim.
# “Hangi mevsimde daha çok kaza oluyor?” sorusuna hızlı bir özet gibi.

season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
}

# month bilgisini mevsime çevirip yeni bir kolon oluşturdum.
df["season"] = df["month"].map(season_map)

# Toplam kazalar içinde mevsimlerin yüzdesini hesapladım.
order = ["Winter", "Spring", "Summer", "Autumn"]
season_counts = df["season"].value_counts(dropna=True)
season_rate = (season_counts / season_counts.sum() * 100).reindex(order)

# Sunumda daha kolay okunması için mevsimlere sabit renk verdim.
season_colors = {
    "Winter": "#2B6CB0",
    "Spring": "#2F855A",
    "Summer": "#D69E2E",
    "Autumn": "#C05621",
}
colors = [season_colors[s] for s in season_rate.index]

plt.figure(figsize=(8,4.5))
bars = plt.bar(season_rate.index, season_rate.values, color=colors,
               edgecolor="#222222", linewidth=0.8)

plt.title("Seasonal Accident Rate (%)", pad=12)
plt.xlabel("Season")
plt.ylabel("Accident Distribution by Season (%)")

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)
plt.ylim(0, season_rate.max() * 1.15)

# Bar üstüne yüzdeleri yazdırma
for b in bars:
    y = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, y + 0.3, f"{y:.1f}%",
             ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("Seasonal Accident Rate (%).png")
plt.show()

print(season_rate.round(2))

# Mevsimlere göre kaza oranlarına baktığımda, kazaların yıl boyunca eşit dağılmadığını görüyorum.
# Kış ayları %23.2 ile en düşük paya sahipken, yaz aylarında (%25.5) ve özellikle sonbaharda (%27.0)
# belirgin bir artış dikkat çekiyor. Bu artışın; sonbaharda hava koşullarının daha değişken olması,
# gün ışığının azalması ve trafik yoğunluğunun artması gibi faktörlerle ilişkili olabileceğini düşünüyorum.


#***** Yıllara Göre Kaza Oranlari *****

sns.barplot(x=df.Year.value_counts().index,y=df.Year.value_counts())
plt.ylabel("Num. of Accidents")
plt.title("Accidents over the years")
plt.savefig("Accidents over the years.png")
plt.show()

print("Mean:{:.2f}   Standard Deviation:{:.2f}".format(df.Year.value_counts().mean(),
                                                     df.Year.value_counts().std()))


#Mean:155689.83   Standard Deviation:14396.02
# Yıllar içinde trafik kazalarının sayısının azaldığını, 2012 yılındaki nadir bir artış dışında aşağı yönlü bir eğilim
# sergilediğini görebiliyoruz. Bunun tam olarak neye bağlı olduğunu anlamak için daha fazla bilgiye ihtiyaç var.
# Belki 2012 Londra Olimpiyatları ve Paralimpik Oyunları bununla ilgili olabilir.
# Bu yıllar boyunca ortalama 155689.83  kaza meydana gelmiş ve Standart Sapma 14396.02 olarak hesaplanmıştır.


#***** Accidents Each Month *****

accidents_peryear = {}
for y in df["Year"].unique():
    accidents_peryear[str(y)] = df[df["Year"] == y]

# 12 farklı ay için 12 farklı renk
month_colors = sns.color_palette("tab20", 12)  # 12 güzel renk
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True,
                        constrained_layout=True, figsize=(15, 9))

year = 2008

for row in range(axs.shape[0]):
    for col in range(axs.shape[1]):
        year += 1

        if str(year) not in accidents_peryear:
            axs[row][col].set_visible(False)
            continue

        data = accidents_peryear[str(year)]

        # Aylık kaza sayıları
        monthly_counts = (
            data.groupby("month_name")["Accident_Index"]
            .count()
            .reset_index()
        )

        # Ay sırasını düzelt
        monthly_counts["month_name"] = pd.Categorical(
            monthly_counts["month_name"], categories=month_order, ordered=True
        )
        monthly_counts = monthly_counts.sort_values("month_name")

        for i, (mn, cnt) in enumerate(zip(monthly_counts["month_name"],
                                          monthly_counts["Accident_Index"])):
            axs[row][col].bar(mn, cnt, color=month_colors[i])

        # Nokta grafiği
        sns.pointplot(ax=axs[row][col],
                      data=monthly_counts,
                      x="month_name", y="Accident_Index",
                      color="black")

        axs[row][col].set_title(str(year))
        axs[row][col].set_ylabel("Num. of Accidents")
        axs[row][col].set_xlabel("Month")
        axs[row][col].tick_params(axis='x', rotation=45)
plt.savefig("Accidents Each Month.png")
fig.show()

#Burada da yine görebiliyoruz ki 2012 yılında genel olarak çok daha fazla hacim var;
# bu bir zirve değil, yalnızca daha yüksek bir toplam. Ancak trendler oldukça benzer görünüyor.
#Ekim ve Kasım ayları, kazaların en fazla yaşandığı aylar olarak sürekli öne çıkıyor.

#***** Kazalarin aylara göre ortalamasi ve standart sapmasi *****

dfmonth = pd.DataFrame(df.groupby("month_name")["Year"].count())
print("\nMean:{:.2f}   Standard Deviation:{:.2f}\n".format(float(dfmonth.mean().unique()), float(dfmonth.std().unique())))
dfmonth.T

#barlarda gördügümüz gibi en cok kaza ekim ve kasim ayinda olmus
#Kötü hava koşulları + gün ışığının azalması + trafik yoğunluğu + saatin geri alınması + kaygan yol koşulları
#indirim aylari(black friday),Halloween – 31 Ekim,Bonfire Night / Guy Fawkes Night – 5 Kasım,Black Friday (Kasım sonu)
#aralik ayindaki noel tatili icin hazirlik,ekimde 1 saat geri alinmasi bu duruma hazirlanmak
df
df.columns


#***** Saat Basina Olusan Kazalar *****
# Saat başına kaza sayısı
hourly_accidents = pd.DataFrame(df.groupby("Hour")["Accident_Index"].count())
hourly_accidents = hourly_accidents.rename({"Accident_Index":"Num. of Accidents"}, axis=1)
top5hours = pd.DataFrame(hourly_accidents["Num. of Accidents"].nlargest(5))
# Diğer 19 saati birleştiriyoruz
elsehours = pd.DataFrame({
    'Hour':'The other 19 hours',
    'Num. of Accidents':[hourly_accidents["Num. of Accidents"].nsmallest(19).sum()]})
elsehours.set_index("Hour", inplace=True)
topvsothers = pd.concat([top5hours, elsehours])

fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15,4))
# 24 saat için 24 farklı renk
bar_colors = sns.color_palette("tab20", len(hourly_accidents))
# Bar plot
axs[0].bar(hourly_accidents.index, hourly_accidents["Num. of Accidents"], color=bar_colors)
axs[0].set_title("Accidents per Hour of the Day")
axs[0].set_xlabel("Hour")
axs[0].set_ylabel("Num. of Accidents")
# Pie chart
cmap = plt.get_cmap("tab20c")
colors = cmap(np.array([1, 2, 5, 6, 9,15]))
axs[1].pie(topvsothers["Num. of Accidents"],
           labels=topvsothers.index,
           autopct='%1.2f%%',
           colors=colors)
axs[1].set_title("Top 5 Hours vs Rest")
plt.savefig("Top 5 Hours vs Rest.png")
plt.show()

# İstatistikler
print("\nMean:{:.2f}   Standard Deviation:{:.2f}\n".format(float(hourly_accidents.mean().unique()),float(hourly_accidents.std().unique())))
print(top5hours, "\n")
hourly_accidents.T

#saat 18.00 ,8.00,15.00,16.00,17.00 bu 5 saat kazalarin yaklasik olarak
#%40 ini olusturuyor.


#***** Gün Bazinda  Accident Severity Incelenmesi *****

daily_accidents = (
    df.groupby("Day_of_Week")["Accident_Index"]
      .count()
      .to_frame("Num. of Accidents")
)

# Gün numaralarını isimlere çevir
day_map = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}
daily_accidents.index = daily_accidents.index.map(day_map)

#  En çok kazanın olduğu ilk 3 gün + diğerleri
top3days = daily_accidents["Num. of Accidents"].nlargest(3)

else_days = pd.DataFrame({
    "Day": "Other 4 days",
    "Num. of Accidents": [
        daily_accidents["Num. of Accidents"].nsmallest(4).sum()
    ]
}).set_index("Day")

topvsothers_days = pd.concat([top3days, else_days])

#  Görselleştirme (bar + pie)
fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15,4))

bar_colors = sns.color_palette("tab10", len(daily_accidents))

# Bar plot
axs[0].bar(
    daily_accidents.index,
    daily_accidents["Num. of Accidents"],
    color=bar_colors
)
axs[0].set_title("Accidents per Day of the Week")
axs[0].set_xlabel("Day")
axs[0].set_ylabel("Num. of Accidents")

# Pie chart
cmap = plt.get_cmap("tab20c")
colors = cmap(np.array([1, 3, 6, 10]))

axs[1].pie(
    topvsothers_days["Num. of Accidents"],
    labels=topvsothers_days.index,
    autopct="%1.2f%%",
    colors=colors
)
axs[1].set_title("Top 3 Days vs Rest")
plt.savefig("Top 3 Days vs Rest.png")
plt.show()

print(
    "\nMean: {:.2f}   Standard Deviation: {:.2f}\n".format(
        float(daily_accidents.mean()),
        float(daily_accidents.std())
    )
)

print("Top 3 Days:\n")
print(top3days)

daily_accidents.T

#----------------- DIŞ KOŞULLAR HAKKINDA ANALİZLER -------------------
#***** Yol Tipini Kazaya Etkisi ******

fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15,4))
road = df["Road_Type"].value_counts()
sns.barplot(ax=axs[0], x = road.index, y = road)
axs[0].set_title("Road types")
axs[1].pie(road, labels=road.index, autopct='%1.2f%%')
axs[1].set_title("Road types")
plt.savefig("Road types.png")
plt.show()

road
#Single carriageway → Tek yönlü yol / Tek şeritli yol
# Dual carriageway → Çift yönlü yol / Bölünmüş yol
# Roundabout → Döner kavşak / Yuvarlak kavşak
# One way street → Tek yönlü sokak / Tek yönlü cadde
# Slip road → Bağlantı yolu / Çıkış-giriş rampası
# Unknown → Bilinmeyen / Tanımlanmamış

#***** Yol Tipi vs Severity *****
# Burada amacım: "Farklı yol tiplerinde kaza şiddeti dağılımı değişiyor mu?"
# Yani örneğin tek şeritli yolda fatal/serious oranı daha mı yüksek, yoksa otoyolda mı?

df.columns = df.columns.str.strip() # Kolon isimlerinde boşluk varsa temizliyorum

sub = df[["Road_Type", "Accident_Severity"]].dropna().copy()
# Accident_Severity bazen string/bozuk gelebiliyor, o yüzden sayıya çevirip hatalıları atıyorum
sub["Accident_Severity"] = pd.to_numeric(sub["Accident_Severity"], errors="coerce")
sub = sub.dropna(subset=["Accident_Severity"])
sub["Accident_Severity"] = sub["Accident_Severity"].astype(int)
sub = sub[sub["Accident_Severity"].isin([1, 2, 3])].copy()

# 1-2-3 değerlerini daha okunur hale getiriyorum
sev_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
sub["Severity"] = sub["Accident_Severity"].map(sev_map)

# çok kalabalık olmasın diye top N
TOP_N = 8
top_types = sub["Road_Type"].value_counts().head(TOP_N).index
sub = sub[sub["Road_Type"].isin(top_types)].copy()

# # Önce sayım tablosu çıkarıyorum: her yol tipinde kaç Fatal/Serious/Slight var?
cnt = pd.crosstab(sub["Road_Type"], sub["Severity"])
pct = cnt.div(cnt.sum(axis=1), axis=0) * 100

# kolonlar garanti + sıra
for c in ["Fatal", "Serious", "Slight"]:
    if c not in pct.columns:
        pct[c] = 0
pct = pct[["Fatal", "Serious", "Slight"]]

# “Severe” diye ek bir metrik çıkarıyorum: Fatal + Serious
# Sunumda "ciddi risk" dediğim kısım aslında burası
pct["Severe"] = pct["Fatal"] + pct["Serious"]

# sırala (Severe düşükten yükseğe; istersen ters çevir)
pct = pct.sort_values("Severe", ascending=True)

# --- Tek grafik: stacked + severe marker ---
labels = pct.index.tolist()
y = np.arange(len(labels))

fatal = pct["Fatal"].values
serious = pct["Serious"].values
slight = pct["Slight"].values
severe = pct["Severe"].values  # nokta konumu

plt.figure(figsize=(12, 5.6))

# stacked barh : her yol tipinde şiddet dağılımını aynı bar üzerinde gösteriyorum
plt.barh(y, fatal, label="Fatal", alpha=0.9)
plt.barh(y, serious, left=fatal, label="Serious", alpha=0.9)
plt.barh(y, slight, left=fatal + serious, label="Slight", alpha=0.9)

# Severe işareti (Fatal+Serious sınırı)
# Bu nokta sağa yaklaştıkça "ciddi kaza oranı artıyor" demek
plt.scatter(severe, y, s=90, marker="D", label="Severe (Fatal+Serious)")

# Severe yüzdesini yaz
for i, v in enumerate(severe):
    plt.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=10)

plt.yticks(y, labels)
plt.xlim(0, 100)
plt.xlabel("Share within road type (%)")
plt.title("Road Type vs Crash Severity (Severe highlighted)")

# Grafiği daha temiz göstermek için üst ve sağ çerçeveyi kaldırıyorum
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.25)

# legend dışarı sağ
plt.legend(title="Legend", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
plt.tight_layout(rect=[0, 0, 0.82, 1])

# Sunumda kullanmak için görseli kaydet
plt.savefig("road_type_single_chart.png", dpi=200, bbox_inches="tight")
plt.show()


#***** Weather Conditions Analysis *****
# Bu bölümde amacım: Hava koşullarına göre kaza şiddeti dağılımı değişiyor mu?
# Yani yağmur, sis gibi durumlarda fatal / serious kazaların oranı artıyor mu, bunu görmek istiyorum.
required_cols = ["Weather_Conditions", "Accident_Severity"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Eksik kolon(lar) var: {missing}")

# Temel temizlik
# Analizde sadece hava koşulu ve kaza şiddeti lazım olduğu için bu iki kolonu alıyorum
sub = df[["Weather_Conditions", "Accident_Severity"]].copy()
sub = sub.dropna(subset=["Weather_Conditions", "Accident_Severity"])

# Accident_Severity bazen string veya bozuk formatta gelebiliyor,
# o yüzden sayıya çevirip çevrilemeyenleri temizliyorum
sub["Accident_Severity"] = pd.to_numeric(sub["Accident_Severity"], errors="coerce")
sub = sub.dropna(subset=["Accident_Severity"])
sub["Accident_Severity"] = sub["Accident_Severity"].astype(int)

# --- En sık görülen hava koşullarını seç ---
# Hava koşulu kategorisi çok fazla olduğu için grafik çok karışmasın diye
# en sık görülen ilk N tanesini alıyorum
TOP_N = 8
top_weather = sub["Weather_Conditions"].value_counts().head(TOP_N).index
sub = sub[sub["Weather_Conditions"].isin(top_weather)].copy()

# --- Sayım tablosu (Weather x Severity) ---
# Her hava koşulunda kaç tane Fatal / Serious / Slight kaza var, bunu sayıyorum
tab = (sub.groupby(["Weather_Conditions", "Accident_Severity"])
         .size()
         .unstack(fill_value=0))

# Bazı hava koşullarında belirli severity seviyeleri hiç olmayabilir,
# grafik bozulmasın diye 1-2-3 kolonlarını garanti altına alıyorum
for s in [1, 2, 3]:
    if s not in tab.columns:
        tab[s] = 0
tab = tab[[1, 2, 3]]

# --- Yüzdeye çevir ---
# Mutlak sayılar yerine oranlara bakmak daha anlamlı,
# çünkü her hava koşulunda toplam kaza sayısı farklı
tab_pct = tab.div(tab.sum(axis=1), axis=0) * 100

# Hava koşullarını en çok kazadan en aza doğru sıralıyorum
# Böylece grafiği soldan sağa okurken mantık bozulmuyor
tab_pct = tab_pct.loc[sub["Weather_Conditions"].value_counts().index]

# --- Etiketleri sadeleştir ---
# Uzun ve karmaşık hava durumu açıklamalarını
# sunumda daha okunur olacak şekilde kısaltıyorum
def short_label(x: str) -> str:
    x = str(x)
    x = x.replace("without high winds", "no winds")
    x = x.replace("with high winds", "high winds")
    x = x.replace("Fog or mist", "Fog/Mist")
    return x

labels = [short_label(i) for i in tab_pct.index]

# Burada her hava koşulu toplamda %100 olacak şekilde,
# kaza şiddetinin dağılımını tek bar üzerinde gösteriyorum
x = np.arange(len(labels))
bottom = np.zeros(len(labels))

plt.figure(figsize=(11, 4.6))

sev_names = {1: "Fatal", 2: "Serious", 3: "Slight"}

for sev in [1, 2, 3]:
    vals = tab_pct[sev].values
    plt.bar(x, vals, bottom=bottom, label=sev_names[sev])
    bottom += vals

# Başlık ve eksenler
plt.title("Accident Severity according to Weather Conditions (%)", pad=12)
plt.xlabel("Weather Forecast (most common)")
plt.ylabel("Rate (%)")

plt.xticks(x, labels, rotation=20, ha="right")
plt.ylim(0, 100)

# Temiz görünüm
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

plt.legend(title="Accident_Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("Accident Severity according to Weather Conditions (%).png")
plt.show()


#****** Light Condietiona Göre ******
# Saat başına kaza sayısı
hourly_accidents = df.groupby("Hour")["Accident_Index"].count().reset_index()
hourly_accidents = hourly_accidents.rename({"Accident_Index": "Num_of_Accidents"}, axis=1)
# Light conditions sayısı
light = df["Light_Conditions"].value_counts()
# Plot
fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15,6))
sns.barplot(ax=axs[0], x = light.index, y = light)
axs[0].set_title("Light conditions")
axs[0].tick_params(labelrotation=70)
sns.barplot(ax=axs[1], x = hourly_accidents["Hour"], y = hourly_accidents["Num_of_Accidents"])
axs[1].set_title("Accidents per Hour")
plt.savefig("Accidents per Hour.png")
plt.show()


light
# Light_Conditions
# Daylight: Street light present               687940
# Darkness: Street lights present and lit      181934
# Darkeness: No street lighting                 48359
# Darkness: Street lighting unknown             11464
# Darkness: Street lights present but unlit      4442

#Burada da görebiliyoruz ki kazaların çoğu gündüz, gün ışığı varken meydana geliyor. Ancak
#kazalarla bir korelasyon olup olmadığını görmek için her ışık koşulunda toplam yolculuk sayısına da ihtiyacımız var.

fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15,6))
speed = df["Speed_limit"].value_counts()
sns.barplot(ax=axs[0], x = speed.index, y = speed)
axs[0].set_title("Speed limit")
axs[1].pie(speed, labels=speed.index, autopct='%1.2f%%')
axs[1].set_title("Speed limit")
plt.savefig("Speed limit.png")
plt.show()
#En çok kazanın meydana geldiği yol bölümleri 30 mil/saat hız sınırına sahip olanlar.
#Daha önce de belirtildiği gibi, korelasyonları incelemek için daha fazla veriye ihtiyacımız var.

speed

#++++++++++++++++++++++++++++++++++++++++yol tipini kazaya etkisi

fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15,4))
road = df["Road_Type"].value_counts()
sns.barplot(ax=axs[0], x = road.index, y = road)
axs[0].set_title("Road types")
axs[1].pie(road, labels=road.index, autopct='%1.2f%%')
axs[1].set_title("Road types")
plt.show()

#+++++++++++++++++++++++++++++hava durumunun kazaya etkisi
fig, axs = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(6,5))
weather = df["Weather_Conditions"].value_counts()
sns.barplot(ax=axs, x = weather.index, y = weather)
axs.set_title("Weather conditions")
axs.tick_params(labelrotation=70)
plt.show()


#****** Yol Tipi, Hava Kosulu Hiz Limiti ve Light Condition Beraber Bakılması *****
sns.set_style("whitegrid")
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(15, 10),constrained_layout=True)
sns.barplot(ax=axs[0, 0],x=road.index,y=road.values,palette="Blues_r")
axs[0, 0].set_title("Road Types", fontsize=13)
sns.barplot(ax=axs[0, 1],x=weather.index,y=weather.values,palette="Greens_r")
axs[0, 1].set_title("Weather Conditions", fontsize=13)
axs[0, 1].tick_params(axis="x", rotation=90)
sns.barplot(ax=axs[1, 0],x=light.index,y=light.values,palette="Oranges_r")
axs[1, 0].set_title("Light Conditions", fontsize=13)
axs[1, 0].tick_params(axis="x", rotation=90)
sns.barplot(ax=axs[1, 1],x=speed.index,y=speed.values,palette="Purples_r")
axs[1, 1].set_title("Speed Limit", fontsize=13)

plt.savefig(
    "Yol_Tipi_Hava_Light_Speed_Renkli.png",
    dpi=300,
    bbox_inches="tight")

plt.show()



###############################
required_cols = ["Light_Conditions", "Accident_Severity"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

sub = df[["Light_Conditions", "Accident_Severity"]].dropna().copy()
sub["Accident_Severity"] = pd.to_numeric(sub["Accident_Severity"], errors="coerce")
sub = sub.dropna(subset=["Accident_Severity"])
sub["Accident_Severity"] = sub["Accident_Severity"].astype(int)

# Top categories (to avoid clutter)
TOP_N = 6
top_light = sub["Light_Conditions"].value_counts().head(TOP_N).index
sub = sub[sub["Light_Conditions"].isin(top_light)].copy()

# --- 2) Pivot -> % within each light condition ---
tab = (sub.groupby(["Light_Conditions", "Accident_Severity"])
         .size()
         .unstack(fill_value=0))

for s in [1, 2, 3]:
    if s not in tab.columns:
        tab[s] = 0
tab = tab[[1, 2, 3]]

tab_pct = tab.div(tab.sum(axis=1), axis=0) * 100


label_map = {
    "Daylight: Street light present": "Daylight",
    "Daylight": "Daylight",
    "Darkness: Street lights present and lit": "Dark: Lights ON",
    "Darkness: Street lights present but unlit": "Dark: Lights OFF",
    "Darkness: No street lighting": "Dark: No lights",
    "Darkness: Street lighting unknown": "Dark: Lighting ?",
}

def short_label(x):
    x = str(x)
    return label_map.get(x, x.replace("Darkness: ", "Dark: ").replace("Daylight: ", "Day: "))

tab_pct.index = [short_label(i) for i in tab_pct.index]

# Optional: nicer order (Daylight first, then darkness variants)
preferred_order = ["Daylight", "Dark: Lights ON", "Dark: Lights OFF", "Dark: No lights", "Dark: Lighting ?"]
tab_pct = tab_pct.reindex([i for i in preferred_order if i in tab_pct.index] + [i for i in tab_pct.index if i not in preferred_order])


y = np.arange(len(tab_pct.index))
left = np.zeros(len(tab_pct.index))

plt.figure(figsize=(11, 5))
sev_names = {1: "Fatal", 2: "Serious", 3: "Slight"}

for sev in [1, 2, 3]:
    vals = tab_pct[sev].values
    plt.barh(y, vals, left=left, label=sev_names[sev])
    left += vals

plt.title("Does Lighting Affect Crash Severity? (Share within each condition, %)", pad=12)
plt.xlabel("Severity share (%)")
plt.ylabel("Light conditions")

plt.yticks(y, tab_pct.index)
plt.xlim(0, 100)

# Clean look
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.25)

# Add Serious+Fatal label at the end of each bar
sf = (tab_pct[1] + tab_pct[2]).values
for i, val in enumerate(sf):
    plt.text(101, i, f"Serious+Fatal: {val:.1f}%", va="center", fontsize=10)

plt.legend(
    title="Severity",
    loc="lower left",
    bbox_to_anchor=(-0.02, -0.22),  # daha sola + biraz aşağı
    ncol=3,
    frameon=True
)
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()
plt.savefig("does_lighting_affect_crash_severity_share.png", dpi=300, bbox_inches="tight")
plt.show()

# ******  Yol Zemini vs Accident Severity *****

# Kolon isimlerinde baş/son boşluk varsa temizle
df.columns = df.columns.str.strip()

# Road surface kolonunu bul
# Önce hem "road" hem "surface" içereni arıyorum, yoksa sadece "surface" bak
candidates = [c for c in df.columns if ("road" in c.lower() and "surface" in c.lower())]
if not candidates:
    candidates = [c for c in df.columns if "surface" in c.lower()]

if not candidates:
    raise ValueError("Road surface kolonu bulunamadı. df.columns çıktısını kontrol etmelisin.")

ROAD_COL = candidates[0]
print("Kullanılan yol yüzeyi kolonu:", ROAD_COL)

# Accident_Severity var mı?
if "Accident_Severity" not in df.columns:
    raise ValueError("Accident_Severity kolonu df içinde yok.")

# Gerekli kolonları alıp temizle
sub = df[[ROAD_COL, "Accident_Severity"]].dropna().copy()

sub["Accident_Severity"] = pd.to_numeric(sub["Accident_Severity"], errors="coerce")
sub = sub.dropna(subset=["Accident_Severity"])
sub["Accident_Severity"] = sub["Accident_Severity"].astype(int)

# sadece 1-2-3 kalsın (bazı datalarda farklı değerler olabiliyor)
sub = sub[sub["Accident_Severity"].isin([1, 2, 3])].copy()

# Çok kalabalık olmasın diye en sık görülen TOP_N yol yüzeyini al
TOP_N = 8
top_surface = sub[ROAD_COL].value_counts().head(TOP_N).index
sub = sub[sub[ROAD_COL].isin(top_surface)].copy()

# Sayım tablosu -> yüzde tablosu (her yol yüzeyi kendi içinde %100)
tab = (sub.groupby([ROAD_COL, "Accident_Severity"])
         .size()
         .unstack(fill_value=0))

# 1,2,3 kolonları garanti olsun
for s in [1, 2, 3]:
    if s not in tab.columns:
        tab[s] = 0
tab = tab[[1, 2, 3]]

pct = tab.div(tab.sum(axis=1), axis=0) * 100

# en sık görülen sıraya göre
order = sub[ROAD_COL].value_counts().index
pct = pct.loc[order]

# Etiketleri biraz kısalt
def short_label(x):
    x = str(x).strip()
    x = x.replace("Wet or damp", "Wet/Damp")
    x = x.replace("Frost or ice", "Frost/Ice")
    x = x.replace("Flood (over 3cm of water)", "Flood (>3cm)")
    x = x.replace("Data missing or out of range", "Missing")
    return x

pct.index = [short_label(i) for i in pct.index]

#Her satırda 3 mini lollipop (Fatal/Serious/Slight)
labels = pct.index.tolist()
y = np.arange(len(labels))

# renkler
colors = {1: "#D64541", 2: "#F39C12", 3: "#27AE60"}   # Fatal / Serious / Slight
names  = {1: "Fatal",   2: "Serious",   3: "Slight"}

# aynı satırda üst üste binmesin diye ufak dikey kaydırma
offsets = {1: -0.22, 2: 0.00, 3: 0.22}

plt.figure(figsize=(12, 5.4))

# her satıra açık bir referans çizgisi (lollipop havası için)
for i in range(len(labels)):
    plt.hlines(i, 0, 100, color="0.92", linewidth=2, zorder=0)

for sev in [1, 2, 3]:
    xvals = pct[sev].values
    yy = y + offsets[sev]

    # çizgi (stem)
    for xi, yi in zip(xvals, yy):
        plt.hlines(yi, 0, xi, color=colors[sev], alpha=0.35, linewidth=6)

    # nokta
    plt.scatter(xvals, yy, s=95, color=colors[sev], label=names[sev], zorder=3)

    # yüzde etiketi
    for xi, yi in zip(xvals, yy):
        plt.text(xi + 0.4, yi, f"{xi:.1f}%", va="center", fontsize=9)

plt.yticks(y, labels)
plt.xlabel("Rate ")
plt.ylabel("Road Surface")
plt.title("Road Surface vs Accident Severity (%)", pad=12)

max_val = pct[[1, 2, 3]].values.max()
plt.xlim(0, max(20, max_val + 4))

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.25)

# legend dışarıda alt kısımda
plt.legend(title="Severity", ncol=3, loc="lower left", bbox_to_anchor=(0.0, -0.22), frameon=True)
plt.subplots_adjust(bottom=0.27)

plt.tight_layout()
plt.savefig("Road Surface vs Accident Severity (%) .png")
plt.show()


# ****** Hız vs Accident Severity ******
# Kolon adlarında boşluk varsa temizliyorum (sonradan isim uyuşmazlığı olmasın diye)
df.columns = df.columns.str.strip()

# Analiz için sadece hız limiti ve kaza şiddeti sütunlarını alıyorum, boşları temizliyorum
sub = df[["Speed_limit", "Accident_Severity"]].dropna().copy()

# Sayısal gibi görünen değerleri gerçekten sayıya çeviriyorum (bozuk olanlar NaN olur)
sub["Speed_limit"] = pd.to_numeric(sub["Speed_limit"], errors="coerce")
sub["Accident_Severity"] = pd.to_numeric(sub["Accident_Severity"], errors="coerce")

# Dönüşüm sonrası oluşan NaN’leri atıyorum
sub = sub.dropna()

# Tipleri int yapıyorum (gruplama daha sağlıklı olsun diye)
sub["Speed_limit"] = sub["Speed_limit"].astype(int)
sub["Accident_Severity"] = sub["Accident_Severity"].astype(int)

# Sadece 1-2-3 sınıflarını (Fatal/Serious/Slight) tutuyorum
sub = sub[sub["Accident_Severity"].isin([1,2,3])]

# Her hız limiti için şiddet sınıflarının sayım tablosunu çıkarıyorum
cnt = (sub.groupby(["Speed_limit","Accident_Severity"])
         .size().unstack(fill_value=0).sort_index())

# Her hız limitindeki toplam kaza sayısı (N)
N = cnt.sum(axis=1)

# Sayımları yüzdeye çeviriyorum: her hız limiti kendi içinde %100 olsun diye
share = cnt.div(N, axis=0) * 100

# Her ihtimale karşı 1-2-3 kolonları eksik kalmasın diye garanti altına alıyorum
for k in [1,2,3]:
    if k not in share.columns:
        share[k] = 0.0
share = share[[1,2,3]]

x = share.index.values  # hız limitleri x ekseni olacak

# Bazı hız limitlerinde veri çok az olabilir → bu noktaları “dikkat” için farklı göstereceğim
LOW_N = 300
is_low = (N < LOW_N).values

plt.figure(figsize=(12,6))

def plot_line(y, label, color):
    # Veri sayısı yeterli olan noktaları normal çiziyorum
    plt.plot(x[~is_low], y[~is_low], marker="o", linewidth=2, label=label, color=color)
    # Veri sayısı az olan noktaları boş daire ile işaretliyorum (yorumu dikkatli yapmak için)
    plt.plot(x[is_low], y[is_low], marker="o", linewidth=0, color=color,
             markerfacecolor="white", markeredgewidth=2, markersize=8)

# Hız limitine göre her şiddet sınıfının yüzde payını çiziyorum
plot_line(share[1].values, "Fatal (1)",  "#d62728")
plot_line(share[2].values, "Serious (2)", "#ff7f0e")
plot_line(share[3].values, "Slight (3)", "#2ca02c")

plt.title("Severity Share by Speed Limit")
plt.xlabel("Speed limit (mph)")
plt.ylabel("Share (%)")
plt.grid(axis="y", alpha=0.25)

# Her hız limitinin altına örnek sayısını da yazıyorum (N) → sunumda çok açıklayıcı oluyor
plt.xticks(x, [f"{v}\nN={int(n)}" for v,n in zip(x, N.values)])

# Legend’i sağa alıyorum, grafiği kalabalık yapmasın diye
plt.legend(title="Severity", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)

# Küçük not: boş dairelerin anlamı (grafik üzerine yapışmasın diye altta)
plt.figtext(0.01, -0.02, f"Not: Boş daire = düşük örnek sayısı (N<{LOW_N})", ha="left", fontsize=10)

plt.tight_layout(rect=[0,0,0.82,1])
plt.savefig("Severity Share by Speed Limit .png")
plt.show()

# İstersem yüzdeleri tablo olarak da görüyorum (kontrol amaçlı)
out = share.copy()
out["N"] = N
out = out.rename(columns={1:"Fatal_%", 2:"Serious_%", 3:"Slight_%"})
print(out.round(2).to_string())


###############################################
# Follium Map (EDA)
###############################################
# Yapacagım eda islemlerinde bazı ciktilari gormek icin df icinde oynamalar yapıcam bu yuzden orjinal veri setinin etkilenmesini istemiyorum
df1 = df.copy()

###############################################
# Mekansal Dagilim - Heatmap (for all)
###############################################
# Tüm kazaların ulke genelinde nerelerde yoğunlaştığını görmek icin
# Kırmızı bolgeler = yuksek yogunluk, yeşil/mavi = dusuk yogunluk.

# Harita için sadece enlem-boylam alıyorum (NaN olan satırlar haritayı bozar)
map_df = df1[["Latitude", "Longitude"]].dropna()

# Veri çok büyük olduğu için harita aşırı yavaşlar.
# Bu yüzden performans amacıyla örnekleme yapıyorum (örnekleme EDA için yeterli).
if len(map_df) > 50000:
    map_df = map_df.sample(50000, random_state=42)

# UK merkezine yakın bir başlangıç haritası oluşturuyorum
# tiles="cartodbpositron" -> daha sade ve “rapor” görünümünde harita teması
m_all = folium.Map(location=[54.5, -3], zoom_start=6, tiles="cartodbpositron")

# HeatMap’e list of [lat, lon] formatında veri vermem gerekiyor
HeatMap(
    data=map_df.values,     # [[lat, lon], [lat, lon], ...]
    radius=8,               # nokta etkisinin yayılma yarıçapı
    blur=15,                # renk geçişlerinin yumuşaklığı
    max_zoom=10             # zoom arttıkça yoğunluğun nasıl davranacağı
).add_to(m_all)

# Görüntülemek için HTML çıktısı kaydediyorum
m_all.save("eda_heatmap_all.html")
print("Kaydedildi: eda_heatmap_all.html")


###############################################
# Mekansal Dagilim - Heatmap (by severity)
###############################################
# 1 (Fatal) ve 2 (Serious) kazaların nerelerde yoğunlaştığını ayrı ayrı görmek.
# - Fatal kazalar sehir disi / yuksek hizli yollarda mı yogun?
# - Serious kazalar hangi bolgelerde daha fazla?

# -------------------------
# FATAL (Accident_Severity = 1)
# -------------------------
# Sadece fatal kazaların koordinatlarını al
fatal_df = df1[df1["Accident_Severity"] == 1][["Latitude", "Longitude"]].dropna()

# Performans icin ornekleme (30000)
if len(fatal_df) > 30000:
    fatal_df = fatal_df.sample(30000, random_state=42)

# Harita olustur
m_fatal = folium.Map(location=[54.5, -3], zoom_start=6, tiles="cartodbpositron")

# Fatal heatmap ekle
HeatMap(
    data=fatal_df.values,
    radius=10,
    blur=20,
    max_zoom=10
).add_to(m_fatal)

# Kaydet
m_fatal.save("eda_heatmap_fatal.html")
print("Kaydedildi: eda_heatmap_fatal.html")

# -------------------------
# SERIOUS (Accident_Severity = 2)
# -------------------------
serious_df = df1[df1["Accident_Severity"] == 2][["Latitude", "Longitude"]].dropna()

if len(serious_df) > 30000:
    serious_df = serious_df.sample(30000, random_state=42)

m_serious = folium.Map(location=[54.5, -3], zoom_start=6, tiles="cartodbpositron")

HeatMap(
    data=serious_df.values,
    radius=10,
    blur=20,
    max_zoom=10
).add_to(m_serious)

m_serious.save("eda_heatmap_serious.html")
print("Kaydedildi: eda_heatmap_serious.html")

###############################################
# Zamana Bagli Heatmap (saat bazli ve animasyon)
###############################################
# Kazaların gun icinde hangi saatlerde yogunlastigini gorsellestirme
# HeatMapWithTime ile saat ilerledikce harita “animasyon” gibi degisecek
# - Rush hour etkisi var mı?
# - Gece saatlerinde yogunluk hangi bolgelerde artiyor?


# Time + koordinatlari al
time_df = df1[["Time", "Latitude", "Longitude"]].dropna()

# Veri buyuk oldugundan orneklem al (200k)
if len(time_df) > 200000:
    time_df = time_df.sample(200000, random_state=42)

# Time kolonu "HH:MM" formatinda oldugundan hour bilgisini cikar
# errors="coerce"  yani bozuk format varsa NaN yapsın, crash olmasın
time_df["Hour"] = pd.to_datetime(time_df["Time"], format="%H:%M", errors="coerce").dt.hour

# Hour NaN olanları at
time_df = time_df.dropna(subset=["Hour"])
time_df["Hour"] = time_df["Hour"].astype(int)

# HeatMapWithTime her “frame” için koordinat listesi ister
# hourly_data[0] -> 00:00 saatine ait noktalar
# hourly_data[23] -> 23:00 saatine ait noktalar
hourly_data = []
for h in range(24):
    subset = time_df[time_df["Hour"] == h][["Latitude", "Longitude"]]
    hourly_data.append(subset.values.tolist())

# Harita olustur
m_time = folium.Map(location=[54.5, -3], zoom_start=6, tiles="cartodbpositron")

# Saatlik animasyon heatmap ekle
HeatMapWithTime(
    data=hourly_data,
    radius=8,
    auto_play=True,     # otomatik oynatsın
    max_opacity=0.8     # yogunluk renginin üst limiti
).add_to(m_time)

# Kaydet
m_time.save("eda_heatmap_time.html")
print("Kaydedildi: eda_heatmap_time.html")


###############################################
# Severity Renkli Nokta Haritasi
###############################################

# Gerekleri sutunlari al ve Nan degerleri cikar
map_df = df1[["Latitude", "Longitude", "Accident_Severity"]].dropna()

n_each = 20000  # Kaldırma durumuna gore arttırabilirim
map_df = (map_df.groupby("Accident_Severity", group_keys=False)
                .apply(lambda x: x.sample(min(len(x), n_each), random_state=42)))

# Harita merkezi (UK)
center = [map_df["Latitude"].mean(), map_df["Longitude"].mean()]
m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

# Renk dict
severity_color = {
    1: "red",     # Fatal
    2: "orange",  # Serious
    3: "green"    # Slight
}

severity_name = {
    1: "Fatal (1)",
    2: "Serious (2)",
    3: "Slight (3)"
}

# Her sinif için ayri layer (istersem tek bir sinif secip gösterebilecegim formatta)
for sev in [1, 2, 3]:
    # Katman grubu
    fg = folium.FeatureGroup(name=severity_name[sev], show=True)

    # Cluster ekleme
    cluster = MarkerCluster().add_to(fg)

    # O sınıfa ait noktalar
    sub = map_df[map_df["Accident_Severity"] == sev]

    # Noktalari ciz
    for lat, lon in sub[["Latitude", "Longitude"]].values:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,                       # nokta boyutu
            color=severity_color[sev],       # kenar rengi
            fill=True,
            fill_color=severity_color[sev],  # ic renk
            fill_opacity=0.6,
            opacity=0.6
        ).add_to(cluster)

    fg.add_to(m)

# Layer kontrol (sag ust konumda secenek kutusu)
folium.LayerControl(collapsed=False).add_to(m)

# Basit legend (sol altta)
legend_html = """
<div style="
position: fixed; 
bottom: 40px; left: 40px; width: 170px; z-index:9999;
background-color: white; padding: 10px; border:2px solid grey; border-radius:6px;
font-size: 14px;
">
<b>Accident Severity</b><br>
<span style="color:red;">●</span> Fatal (1)<br>
<span style="color:orange;">●</span> Serious (2)<br>
<span style="color:green;">●</span> Slight (3)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Kaydet
m.save("eda_severity_colored_points.html")
print("Kaydedildi: eda_severity_colored_points.html")


###############################################
# Mekansal Hotspot - DBSCAN Clustering
###############################################
# Heatmap sadece yogunlugu boyuyor fakat DBSCAN ise kumeler bulur ve merkezlerini gosterir

EARTH_RADIUS_KM = 6371.0088

def dbscan_hotspots(df1, eps_km=1.0, min_samples=120, severity=None, sample_n=200000):
    w = df1[["Latitude", "Longitude", "Accident_Severity"]].dropna().copy()

    # severity filtresi
    if severity is not None:
        w = w[w["Accident_Severity"] == severity]

    # Performans için ornekleme
    if sample_n is not None and len(w) > sample_n:
        w = w.sample(sample_n, random_state=42)

    # Derece -> radyan (haversine icin sart)
    coords_rad = np.radians(w[["Latitude", "Longitude"]].values)

    eps_rad = eps_km / EARTH_RADIUS_KM

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", n_jobs=-1)
    labels = db.fit_predict(coords_rad)

    w["cluster"] = labels

    clustered = w[w["cluster"] != -1].copy()

    summary = (clustered.groupby("cluster")
               .agg(count=("cluster", "size"),
                    lat=("Latitude", "mean"),
                    lon=("Longitude", "mean"))
               .sort_values("count", ascending=False)
               .reset_index())

    return w, summary


def plot_hotspot_centers(summary, file_name="eda_dbscan_hotspots.html", zoom=6):
    center = [summary["lat"].mean(), summary["lon"].mean()]
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")

    for _, row in summary.iterrows():
        popup = f"Cluster: {int(row['cluster'])}<br>Count: {int(row['count'])}"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="red",
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=250)
        ).add_to(m)

    m.save(file_name)
    print("Kaydedildi:", file_name)
    return m


#  Tum kazalar icin hotspot merkezleri
_, summary_all = dbscan_hotspots(df, eps_km=1.0, min_samples=120, severity=None, sample_n=200000)
print("Hotspot sayısı (ALL):", summary_all.shape[0])
m_hot_all = plot_hotspot_centers(summary_all, file_name="eda_dbscan_hotspots_all.html", zoom=6)

# Fatal icin ayri (fatal az olduğu icin esikler daha dusuk)
_, summary_fatal = dbscan_hotspots(df, eps_km=2.0, min_samples=15, severity=1, sample_n=80000)
print("Hotspot sayısı (FATAL):", summary_fatal.shape[0])
m_hot_fatal = plot_hotspot_centers(summary_fatal, file_name="eda_dbscan_hotspots_fatal.html", zoom=6)


###############################################
# DBSCAN Hotspot + Count label
###############################################
EARTH_RADIUS_KM = 6371.0088

def dbscan_hotspots_summary(df, eps_km=1.0, min_samples=120, severity=None,
                           sample_n=200000, random_state=42):
    """
    df parametresi üzerinden çalışır (global df1 kullanmaz).
    DBSCAN ile cluster merkezlerini ve cluster büyüklüklerini döndürür.
    """

    w2 = df[["Latitude", "Longitude", "Accident_Severity"]].dropna().copy()

    # severity filtresi
    if severity is not None:
        w2 = w2[w2["Accident_Severity"] == severity]

    # performans için örnekleme
    if sample_n is not None and len(w2) > sample_n:
        w2 = w2.sample(sample_n, random_state=random_state)

    # haversine için derece -> radyan
    coords_rad = np.radians(w2[["Latitude", "Longitude"]].values)

    eps_rad = eps_km / EARTH_RADIUS_KM
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", n_jobs=-1)
    labels = db.fit_predict(coords_rad)

    w2["cluster"] = labels

    clustered = w2[w2["cluster"] != -1].copy()
    summary = (clustered.groupby("cluster")
               .agg(count=("cluster", "size"),
                    lat=("Latitude", "mean"),
                    lon=("Longitude", "mean"))
               .sort_values("count", ascending=False)
               .reset_index())

    return summary

def scale_radius(count, c_min, c_max, r_min=4, r_max=18):
    """
    count büyük oldukca marker buyuyecek
    sqrt ile asıro büyük farkları yumusat
    """
    if c_max == c_min:
        return (r_min + r_max) / 2

    # sqrt olcek
    x = (np.sqrt(count) - np.sqrt(c_min)) / (np.sqrt(c_max) - np.sqrt(c_min))
    return r_min + x * (r_max - r_min)


def add_hotspot_layer(m, summary, layer_name, color, top_n=60, show=True):
    fg = folium.FeatureGroup(name=layer_name, show=show)

    if summary is None or summary.empty:
        fg.add_to(m)
        return

    # Fazla nokta olursa okunamayacagı icin top_n
    summary_plot = summary.head(top_n).copy()

    c_min, c_max = summary_plot["count"].min(), summary_plot["count"].max()

    for _, row in summary_plot.iterrows():
        lat, lon, cnt = float(row["lat"]), float(row["lon"]), int(row["count"])
        r = scale_radius(cnt, c_min, c_max, r_min=5, r_max=20)

        # Hotspot merkezi (boyut=count)
        folium.CircleMarker(
            location=[lat, lon],
            radius=r,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.45,
            opacity=0.8,
            tooltip=f"{layer_name} | Count: {cnt}"
        ).add_to(fg)

        # Ustune sayısını yaz (renk sınıf rengiyle)
        folium.Marker(
            location=[lat, lon],
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(0, 0),
                html=f"""
                <div style="
                    font-size: 13px;
                    font-weight: 700;
                    color: {color};
                    text-shadow: -1px 0 white, 0 1px white, 1px 0 white, 0 -1px white;
                ">{cnt}</div>
                """
            )
        ).add_to(fg)

    fg.add_to(m)


# Severity ayar (fatal az oldugu icin esik daha gevsek)
sev_cfg = {
    1: {"name": "Fatal (1) Hotspots",   "color": "red",    "eps_km": 2.0, "min_samples": 15,  "sample_n": 80000,  "top_n": 60},
    2: {"name": "Serious (2) Hotspots", "color": "orange", "eps_km": 1.5, "min_samples": 40,  "sample_n": 150000, "top_n": 60},
    3: {"name": "Slight (3) Hotspots",  "color": "green",  "eps_km": 1.0, "min_samples": 120, "sample_n": 200000, "top_n": 60},
}

# Harita merkezi
center = [df1["Latitude"].mean(), df1["Longitude"].mean()]
m_hot = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")

# Her severity için layer ekle
for sev in [1, 2, 3]:
    cfg = sev_cfg[sev]
    summary = dbscan_hotspots_summary(
        df,
        eps_km=cfg["eps_km"],
        min_samples=cfg["min_samples"],
        severity=sev,
        sample_n=cfg["sample_n"]
    )
    add_hotspot_layer(
        m_hot,
        summary=summary,
        layer_name=cfg["name"],
        color=cfg["color"],
        top_n=cfg["top_n"],
        show=True
    )
    print(f"{cfg['name']} -> hotspot sayısı:", summary.shape[0])

folium.LayerControl(collapsed=False).add_to(m_hot)

m_hot.save("eda_dbscan_hotspots_labeled_by_severity.html")
print("Kaydedildi: eda_dbscan_hotspots_labeled_by_severity.html")
m_hot


###########################################
# DATA PREPROCESSING
###########################################

def grab_col_names(dataframe, cat_th=5, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Observations: 934089
# Variables: 39
# cat_cols: 15
# num_cols: 19
# cat_but_car: 5
# num_but_cat: 3
###############################################
# KATEGORIK DEGISKEN ANALIZI
###############################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)


###############################################
# SAYISAL DEGISKEN ANALIZI
###############################################

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(f"\n### Numerical Summary: {numerical_col} ###")
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.ylabel("Count")
        plt.title(f"{numerical_col} Histogram")
        plt.show(block=True)

    print("########################################")


for col in num_cols:
    num_summary(df, col, plot=True)


print(cat_cols)


######################################
# HEDEF DEGISKEN ANALIZI
######################################
def target_distribution(dataframe, target):
    print("TARGET VALUE COUNTS")
    print(dataframe[target].value_counts(), end="\n\n")

    print("TARGET RATIO (%)")
    print((100 * dataframe[target].value_counts() / len(dataframe)).round(2))


target_distribution(df, "Accident_Severity")

#*********************************************************
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ(Analysis of Numerical Variable by the Target )
#*********************************************************


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Accident_Severity", col)


#*********************************************************
# KATEGORIK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ(Analysis of Categorical Variable by the Target )
#*********************************************************

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Accident_Severity", col)


#*********************************************************
# KORELASYON(Analysis of Correlation)
#*********************************************************

def correlation_matrix(df, cols):
    corr = df[cols].corr()
    fig = plt.gcf()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(),mask=mask, annot=True, linewidths=0.3, fmt=".2f", annot_kws={'size': 10}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

# Sayısal değişkenkerin birbirleri ile korelasyonu

correlation_matrix(df, num_cols)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    numeric_df = dataframe.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
#['Longitude', 'Latitude', 'Local_Authority_(District)', 'year'] en son silinecek
drop_list = high_correlated_cols(df)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1))


######################################
# Eksik Deger Analizi
######################################
df.isnull().sum().max()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

msno.matrix(df)
plt.title("Missing Data Matrix", fontsize=14)
plt.savefig("Missing Data Matrix.png")
plt.show()

# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)# _NA_ (1 = eksik, 0 = eksik değil)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Accident_Severity", na_columns)

# 1. Drop columns with more than 90% missing values
df.drop(columns=['Special_Conditions_at_Site','Carriageway_Hazards','Junction_Detail'], inplace=True)

df['Junction_Control'].fillna('Unknown', inplace=True)
df['LSOA_of_Accident_Location'].fillna('Unknown', inplace=True)
df['Pedestrian_Crossing-Human_Control'].fillna('Unknown', inplace=True)
df['Pedestrian_Crossing-Physical_Facilities'].fillna('Unknown', inplace=True)
df['Weather_Conditions'].fillna('Unknown', inplace=True)
df['Road_Surface_Conditions'].fillna('Unknown', inplace=True)
df['Did_Police_Officer_Attend_Scene_of_Accident'].fillna('Unknown', inplace=True)

print(df.columns.tolist())

df.dropna(subset=["year","month","day_name","month_name","Hour"],inplace=True)

######################################
# Aykırı Deger Analizi
######################################



def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "Accident_Severity":
      print(col, check_outlier(df, col))


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "Accident_Severity":
        replace_with_thresholds(df,col)

for col in num_cols:
    if col != "Accident_Severity":
      print(col, check_outlier(df, col))


# Drop/fill sonrası kolon listelerini GÜNCELLE
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Observations: 934089
# Variables: 36
# cat_cols: 14
# num_cols: 17
# cat_but_car: 5
# num_but_cat: 4

# target asla feature listelerinde olmasın
cat_cols = [c for c in cat_cols if c != "Accident_Severity"]
num_cols = [c for c in num_cols if c != "Accident_Severity"]

# eksik kolon listesini de yeniden al
na_columns = missing_values_table(df, na_name=True)
print(na_columns)



###############################################
# FEATURE ENGINEERING (FINAL VERSION)
###############################################
# Amaç:
# - Gerçek hayattaki risk senaryolarını yakalamak
# - Hedef değişkenle anlamlı ilişki kuran feature’lar üretmek
# - Multicollinearity (VIF) yaratmamak
###############################################

# -------------------------------------------------
# 1) TIME BASED FEATURES
# -------------------------------------------------

# Weekend (hafta sonu davranışı farklı olabilir)
df["NEW_Is_Weekend"] = df["Day_of_Week"].isin([1, 7]).astype(int)

df["NEW_Late_Night"] = df["Hour"].between(0, 4).astype(int)


# -------------------------------------------------
# 2) URBAN / RURAL
# -------------------------------------------------

# Kırsal alan (yüksek hız + geç müdahale etkisi)
df["NEW_Is_Rural"] = (df["Urban_or_Rural_Area"].astype(str) == "2").astype(int)

# -------------------------------------------------
# 3) ENVIRONMENTAL RISK (TEK VE ÖZET)
# -------------------------------------------------
# Gece durumu
df["NEW_Is_Dark"] = df["Light_Conditions"].astype(str).str.contains(
    "dark", case=False, na=False).astype(int)

# Ağır hava koşulları
bad_weather = [
    "Raining with high winds",
    "Snowing without high winds",
    "Snowing with high winds",
    "Fog or mist"]
df["NEW_Bad_Weather"] = df["Weather_Conditions"].isin(bad_weather).astype(int)

# Kötü yol yüzeyi
bad_surface = ["Wet/Damp", "Snow", "Frost/Ice", "Flood (Over 3cm of water)"]
df["NEW_Bad_Road_Surface"] = df["Road_Surface_Conditions"].isin(bad_surface).astype(int)

# Birleşik çevresel risk (binary – skor değil)
df["NEW_Environmental_Risk"] = (
    (df["NEW_Is_Dark"] == 1) |
    (df["NEW_Bad_Weather"] == 1) |
    (df["NEW_Bad_Road_Surface"] == 1)).astype(int)

# Sokak aydınlatması yok (en güçlü tekil feature’lardan biri)
df["NEW_No_Street_Lighting"] = df["Light_Conditions"].astype(str).str.contains(
    "no street lighting", case=False, na=False).astype(int)

# -------------------------------------------------
# 4) ROAD & JUNCTION
# -------------------------------------------------

# Kontrolsüz kavşak + ikinci yol varlığı
df["NEW_Junction_Uncontrolled"] = (
    (df["2nd_Road_Class"] != -1) &
    (df["Junction_Control"] == "Giveway or uncontrolled")).astype(int)


# -------------------------------------------------
# 5) PEDESTRIAN RISK
# -------------------------------------------------

# Şehir içi + yaya altyapısı yok → riskli senaryo
df["NEW_Ped_Risk_Proxy"] = (
    (df["NEW_Is_Rural"] == 0) &
    (df["Pedestrian_Crossing-Physical_Facilities"]
     == "No physical crossing within 50 meters")).astype(int)

# -------------------------------------------------
# 6) STRONG INTERACTION
# -------------------------------------------------

# Tek şerit + yüksek hız → fatal oranı ciddi artıyor
df["NEW_Speed_Road_Mismatch"] = (
    (df["Road_Type"] == "Single carriageway") &
    (df["Speed_limit"] >= 60)).astype(int)


# VIF'te kullanılacak FINAL feature set
vif_features = [
    "Speed_limit",
    "Hour",
    "NEW_Is_Weekend",
    "NEW_Late_Night",
    "NEW_Is_Rural",
    "NEW_Environmental_Risk",
    "NEW_No_Street_Lighting",
    "NEW_Junction_Uncontrolled",
    "NEW_Ped_Risk_Proxy",
    "NEW_Speed_Road_Mismatch"]

# Güvenli seçim: df'de gerçekten var olanları al
vif_features_exist = [c for c in vif_features if c in df.columns]

X = df[vif_features_exist].copy()
X = X.dropna()

# Sabit terim
X_const = add_constant(X)

# VIF hesaplama
vif_df = pd.DataFrame({
    "Feature": X_const.columns,
    "VIF": [
        variance_inflation_factor(X_const.values, i)
        for i in range(X_const.shape[1])]})

print("\nVIF RESULTS")
print(vif_df.to_string(index=False))

# ----------------------------------------------------------


# NEW_ feature'ları al
new_features = [c for c in df.columns if c.startswith("NEW_")]

target = "Accident_Severity"

for col in new_features:
    print("\n" + "=" * 60)
    print(f"CROSSTAB FOR: {col}")
    print("=" * 60)

    # Normalize edilmiş crosstab (oranlar)
    ct = pd.crosstab(
        df[col],
        df[target],
        normalize="index"
    )

    print(ct.round(3))



"""
VIF RESULTS
                  Feature    VIF
                    const 31.037
              Speed_limit  2.280
                     Hour  1.494
           NEW_Is_Weekend  1.034
           NEW_Late_Night  1.621
             NEW_Is_Rural  3.043
   NEW_Environmental_Risk  1.166
   NEW_No_Street_Lighting  1.261
NEW_Junction_Uncontrolled  1.077
       NEW_Ped_Risk_Proxy  2.049
  NEW_Speed_Road_Mismatch  1.634
============================================================
CROSSTAB FOR: NEW_Is_Weekend
============================================================
Accident_Severity     1     2     3
NEW_Is_Weekend                     
0                 0.010 0.133 0.856
1                 0.016 0.158 0.825
============================================================
CROSSTAB FOR: NEW_Late_Night
============================================================
Accident_Severity     1     2     3
NEW_Late_Night                     
0                 0.011 0.137 0.853
1                 0.031 0.196 0.774
============================================================
CROSSTAB FOR: NEW_Is_Rural
============================================================
Accident_Severity     1     2     3
NEW_Is_Rural                       
0                 0.006 0.126 0.868
1                 0.022 0.165 0.814
============================================================
CROSSTAB FOR: NEW_Is_Dark
============================================================
Accident_Severity     1     2     3
NEW_Is_Dark                        
0                 0.009 0.133 0.858
1                 0.018 0.158 0.825
============================================================
CROSSTAB FOR: NEW_Bad_Weather
============================================================
Accident_Severity     1     2     3
NEW_Bad_Weather                    
0                 0.012 0.140 0.848
1                 0.012 0.122 0.865
============================================================
CROSSTAB FOR: NEW_Bad_Road_Surface
============================================================
Accident_Severity        1     2     3
NEW_Bad_Road_Surface                  
0                    0.011 0.142 0.846
1                    0.012 0.132 0.855
============================================================
CROSSTAB FOR: NEW_Environmental_Risk
============================================================
Accident_Severity          1     2     3
NEW_Environmental_Risk                  
0                      0.010 0.137 0.853
1                      0.014 0.142 0.844
============================================================
CROSSTAB FOR: NEW_No_Street_Lighting
============================================================
Accident_Severity          1     2     3
NEW_No_Street_Lighting                  
0                      0.010 0.136 0.854
1                      0.041 0.196 0.764
============================================================
CROSSTAB FOR: NEW_Junction_Uncontrolled
============================================================
Accident_Severity             1     2     3
NEW_Junction_Uncontrolled                  
0                         0.016 0.150 0.834
1                         0.007 0.128 0.864
============================================================
CROSSTAB FOR: NEW_Ped_Risk_Proxy
============================================================
Accident_Severity      1     2     3
NEW_Ped_Risk_Proxy                  
0                  0.017 0.154 0.829
1                  0.006 0.124 0.870
============================================================
CROSSTAB FOR: NEW_Speed_Road_Mismatch
============================================================
Accident_Severity           1     2     3
NEW_Speed_Road_Mismatch                  
0                       0.009 0.130 0.862
1                       0.031 0.204 0.765
"""
df.shape
#(934089, 48)

##################################################################
# RARE ENCODING
##################################################################
def rare_analyser(dataframe, target, cat_cols, positive_class=None):
    """
    positive_class:
      - None ise: target ortalamasını basar (target 1/2/3 ise ortalama çok anlamlı olmayabilir)
      - 1 veya 2 gibi verirsen: o sınıfı 1 yapıp  oranı basar
    """
    df_ = dataframe.copy()

    # hedefi binary'e çevir (fatal için positive_class=1,...)
    if positive_class is not None:
        df_["_target_bin"] = (df_[target] == positive_class).astype(int)
        tcol = "_target_bin"
        tname = f"{target}=={positive_class} rate"
    else:
        tcol = target
        tname = "TARGET_MEAN"

    for col in cat_cols:
        vc = df_[col].value_counts(dropna=False)
        n_unique = vc.shape[0]
        print(f"\n{col} : {n_unique} unique")

        out = pd.DataFrame({
            "COUNT": vc,
            "RATIO": (vc / len(df_)).round(4),
        })

        # hedef ortalaması / oranı
        out[tname] = df_.groupby(col)[tcol].mean().round(4)

        print(out)

# Fatal oranı (1 = Fatal)
rare_analyser(df, target="Accident_Severity", cat_cols=cat_cols, positive_class=1)

# Serious oranı (2 = Serious)
rare_analyser(df, target="Accident_Severity", cat_cols=cat_cols, positive_class=2)

# Serious oranı (3)
rare_analyser(df,target="Accident_Severity", cat_cols=cat_cols, positive_class=3)


def rare_encoder(dataframe, rare_perc=0.01, cat_cols=None, verbose=True):
    """
    rare_perc: 0.01 => %1'in altındakiler Rare
    cat_cols: None verirsen object+category kolonları otomatik seçer
    verbose: True ise hangi kolonlarda kaç sınıf Rare oldu yazar

    return:
      df_out, rare_info
      rare_info[col] = Rare yapılan sınıfların listesi
    """
    df_out = dataframe.copy()

    # kategorikleri seç (int görünümlü kategorikler sende ayrı listede olabilir; cat_cols verirsen onu kullan)
    if cat_cols is None:
        cat_cols = [c for c in df_out.columns if df_out[c].dtype.name in ["object", "category"]]

    rare_info = {}

    n = len(df_out)

    for col in cat_cols:
        ratios = df_out[col].value_counts(dropna=False) / n
        rare_labels = ratios[ratios < rare_perc].index

        if len(rare_labels) > 0:
            rare_info[col] = list(rare_labels)
            df_out[col] = np.where(df_out[col].isin(rare_labels), "Rare", df_out[col])

            if verbose:
                print(f"{col}: {len(rare_labels)} sınıf Rare yapıldı. (eşik={rare_perc})")

    return df_out, rare_info

df_rare, rare_info = rare_encoder(df, rare_perc=0.01, cat_cols=cat_cols)

for col, labels in rare_info.items():
    print(f"\n--- {col} ---")
    print("Rare yapılan sınıflar:", labels)
    print("Rare count:", (df_rare[col] == "Rare").sum())

##### Cikti sonucu

##################################################################
# LABEL & ONE HOT ENCODING
##################################################################
target ="Accident_Severity"
target
cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols = [c for c in cat_cols if c != target]
num_cols = [c for c in num_cols if c != target]
df.head()


def label_encoder(dataframe, col):
    le = LabelEncoder()
    dataframe[col] = le.fit_transform(dataframe[col].astype(str))
    return dataframe

# 1) Binary kategorikler: sadece object/category olup 2 sınıflı olanlar
binary_cols = [c for c in cat_cols
               if df[c].dtype.name in ["object", "category"] and df[c].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# 2) One-Hot: binary olmayan kategorikler
ohe_cols = [c for c in cat_cols if c not in binary_cols]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

df = one_hot_encoder(df, ohe_cols, drop_first=True)

print("Binary encoded cols:", binary_cols)
print("OHE cols count:", len(ohe_cols))
print("Final shape:", df.shape)

# Son kontrol: NA kaldı mı?
print("\nRemaining NA total:", df.isnull().sum().sum())
df.filter(like="Urban_or_Rural_Area").head()

"""
Observations: 934089
Variables: 47
cat_cols: 25
num_cols: 17
cat_but_car: 5
num_but_cat: 15
Binary encoded cols: []
OHE cols count: 24
Final shape: (934089, 93)
Remaining NA total: 0
"""

#######################################################################################################################
#         MODELLEME
#######################################################################################################################
len(df.columns)
#96
# ['Accident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR',
#        'Longitude', 'Latitude', 'Police_Force', 'Accident_Severity', 'Date',
#        'Day_of_Week', 'Time', 'Local_Authority_(District)',
#        'Local_Authority_(Highway)', '1st_Road_Class', '1st_Road_Number',
#        'Speed_limit', '2nd_Road_Class', '2nd_Road_Number',
#        'LSOA_of_Accident_Location', 'Year', 'Timestamp', 'year', 'month',
#        'Hour', 'Road_Type_One way street', 'Road_Type_Roundabout',
#        'Road_Type_Single carriageway', 'Road_Type_Slip road',
#        'Road_Type_Unknown', 'Junction_Control_Automatic traffic signal',
#        'Junction_Control_Giveway or uncontrolled',
#        'Junction_Control_Stop Sign', 'Junction_Control_Unknown',
#        'Pedestrian_Crossing-Human_Control_Control by school crossing patrol',
#        'Pedestrian_Crossing-Human_Control_None within 50 metres',
#        'Pedestrian_Crossing-Physical_Facilities_Footbridge or subway',
#        'Pedestrian_Crossing-Physical_Facilities_No physical crossing within 50 meters',
#        'Pedestrian_Crossing-Physical_Facilities_Pedestrian phase at traffic signal junction',
#        'Pedestrian_Crossing-Physical_Facilities_Zebra crossing',
#        'Pedestrian_Crossing-Physical_Facilities_non-junction pedestrian crossing',
#        'Light_Conditions_Darkness: Street lighting unknown',
#        'Light_Conditions_Darkness: Street lights present and lit',
#        'Light_Conditions_Darkness: Street lights present but unlit',
#        'Light_Conditions_Daylight: Street light present',
#        'Weather_Conditions_Fine without high winds',
#        'Weather_Conditions_Fog or mist', 'Weather_Conditions_Other',
#        'Weather_Conditions_Raining with high winds',
#        'Weather_Conditions_Raining without high winds',
#        'Weather_Conditions_Snowing with high winds',
#        'Weather_Conditions_Snowing without high winds',
#        'Weather_Conditions_Unknown',
#        'Road_Surface_Conditions_Flood (Over 3cm of water)',
#        'Road_Surface_Conditions_Frost/Ice', 'Road_Surface_Conditions_Snow',
#        'Road_Surface_Conditions_Unknown', 'Road_Surface_Conditions_Wet/Damp',
#        'Did_Police_Officer_Attend_Scene_of_Accident_Unknown',
#        'Did_Police_Officer_Attend_Scene_of_Accident_Yes', 'day_name_Monday',
#        'day_name_Saturday', 'day_name_Sunday', 'day_name_Thursday',
#        'day_name_Tuesday', 'day_name_Wednesday', 'month_name_August',
#        'month_name_December', 'month_name_February', 'month_name_January',
#        'month_name_July', 'month_name_June', 'month_name_March',
#        'month_name_May', 'month_name_November', 'month_name_October',
#        'month_name_September', 'season_Spring', 'season_Summer',
#        'season_Winter', 'Number_of_Vehicles_2.0', 'Number_of_Vehicles_3.0',
#        'Number_of_Vehicles_3.5', 'Number_of_Casualties_2.0',
#        'Number_of_Casualties_3.0', 'Number_of_Casualties_3.5',
#        'Urban_or_Rural_Area_2', 'NEW_Is_Weekend_1', 'NEW_Late_Night_1',
#        'NEW_Is_Rural_1', 'NEW_Is_Dark_1', 'NEW_Bad_Weather_1',
#        'NEW_Bad_Road_Surface_1', 'NEW_Environmental_Risk_1',
#        'NEW_No_Street_Lighting_1', 'NEW_Junction_Uncontrolled_1',
#        'NEW_Ped_Risk_Proxy_1', 'NEW_Speed_Road_Mismatch_1'],
#       dtype='object')




drop_liste = [

    # ID / Ezberleme
    "Accident_Index",


    # LEAKAGE – sonuçtan gelen bilgiler
    "Number_of_Casualties_2.0",
    "Number_of_Casualties_3.0",
    "Number_of_Casualties_3.5",
    "Number_of_Vehicles_2.0",
    "Number_of_Vehicles_3.0",
    "Number_of_Vehicles_3.5",
    "Did_Police_Officer_Attend_Scene_of_Accident_Yes",
    "Did_Police_Officer_Attend_Scene_of_Accident_Unknown",

    # Zaman – redundant / duplicate
    "Date",
    "Timestamp",
    "year",          # duplicate
    "month",         # duplicate
    "Time",
    "Day_of_Week",

    # Konum – aşırı spesifik (overfitting)
    "Location_Easting_OSGR",
    "Location_Northing_OSGR",
    "Local_Authority_(Highway)",
    "Latitude",
    "Longitude",
    "LSOA_of_Accident_Location"
]

###################### LGBMClassifier ILE MODELLEME ###################################################################
#++++++++++++++ Bagimli ve Bagimsiz degiskenlerin ayarlanmasi

y = (df["Accident_Severity"] == 1).astype(int)
X = df.drop(columns=["Accident_Severity"] + drop_liste)

#Xin uygun fromata getirilmesi


def clean_feature_names(df):
    df = df.copy()
    df.columns = [
        re.sub(r'[^A-Za-z0-9_]', '_', col)
        for col in df.columns
    ]
    return df

X = clean_feature_names(X)
y.shape
# (934089,)
X.shape
#(934089, 74)

#+++++++++++++++Stratified CV + scale_pos_weight HESAPLA

# sınıf oranına göre
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
scale_pos_weight
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

lgbm_model = LGBMClassifier(
    objective="binary",
    random_state=17,
    scale_pos_weight=scale_pos_weight,  #  EN KRİTİK
    n_estimators=500,
    learning_rate=0.05,
    colsample_bytree=0.8,
    subsample=0.8,
    min_child_samples=100,
    n_jobs=-1)

cv_results = cross_validate(
    lgbm_model,
    X, y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    n_jobs=-1)

cv_results["test_accuracy"].mean()#--np.float64(0.7540009562559238)
cv_results["test_precision"].mean()#---np.float64(0.03140614568472434)
cv_results["test_recall"].mean()#----np.float64(0.6736477819494482)
cv_results["test_f1"].mean()#------np.float64(0.06001411040622523)
cv_results["test_roc_auc"].mean()#-----np.float64(0.7844232611210022)


#+++++++++++++++++++++++Hiperparametre Optimizasyonu

lgbm_params = {
    "learning_rate": [0.03, 0.05],
    "n_estimators": [500, 800],
    "colsample_bytree": [0.7, 0.8],
    "min_child_samples": [100, 200]}

lgbm_best_grid = GridSearchCV(
    lgbm_model,
    lgbm_params,
    cv=cv,
    scoring="f1",        #  imbalance için
    n_jobs=-1,
    verbose=2).fit(X, y)

lgbm_best_grid.best_params_
# {'colsample_bytree': 0.8,
#  'learning_rate': 0.05,
#  'min_child_samples': 200,
#  'n_estimators': 800}

#Hiperparametre Optimizasyon sonucu model kurulumu

lgbm_final = LGBMClassifier(
    **lgbm_best_grid.best_params_,
    objective="binary",
    scale_pos_weight=scale_pos_weight,
    random_state=17,
    n_jobs=-1)

cv_results_final = cross_validate(
    lgbm_final,
    X, y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    n_jobs=-1)

cv_results_final["test_accuracy"].mean()#----np.float64(0.7768017811608284)
cv_results_final["test_precision"].mean()#-----np.float64(0.033460496591723635)
cv_results_final["test_recall"].mean()#------np.float64(0.6507761458068385)
cv_results_final["test_f1"].mean()#------np.float64(0.06364821167057455)
cv_results_final["test_roc_auc"].mean()#------np.float64(0.7844686176415042)



##+++++++++++++++++++++++TRESHOLD TUNNING

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=17)

lgbm_final.fit(X_train, y_train)

y_prob = lgbm_final.predict_proba(X_test)[:, 1]


#+++++++++++++++++++++TRESHOLD TUNNING


for t in [0.03, 0.05, 0.1, 0.15, 0.2]:
    y_pred = (y_prob >= t).astype(int)
    print(
        f"t={t:.2f} | "
        f"Precision={precision_score(y_test, y_pred):.3f} | "
        f"Recall={recall_score(y_test, y_pred):.3f} | "
        f"F1={f1_score(y_test, y_pred):.3f}"
    )


"""
[LightGBM] [Info] Start training from score -4.440232
t=0.03 | Precision=0.012 | Recall=0.997 | F1=0.023
t=0.05 | Precision=0.012 | Recall=0.994 | F1=0.024
t=0.10 | Precision=0.013 | Recall=0.975 | F1=0.026
t=0.15 | Precision=0.014 | Recall=0.946 | F1=0.028
t=0.20 | Precision=0.016 | Recall=0.908 | F1=0.031
"""
#SUNUMDA AYNEN ŞUNU SÖYLE

#“Recall ve precision arasındaki denge incelendiğinde,
#threshold=0.20, %90’ın üzerinde fatal yakalama oranı sağlarken
#yanlış alarm oranını görece azaltması nedeniyle tercih edilmiştir.”



#CONFUSING MATRIX ILE IS KARARI
# threshold
t = 0.20

# tahmin
y_pred = (y_prob >= t).astype(int)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

# görselleştirme
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Severe (0)", "Severe (1)"])

plt.figure(figsize=(6, 5))
disp.plot(values_format="d")
plt.title("LightGBM Confusion Matrix (Threshold = 0.20)")
plt.savefig("lightgbm_confusion_matrix 20.png")
plt.show()



##+++++++++++++++FEATURE IMPORTANCE
booster = lgbm_final.booster_  # model fit edildikten sonra oluşur

feature_importance = pd.DataFrame({
    "feature": booster.feature_name(),
    "importance": booster.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False).reset_index(drop=True)

top_n = 20
fi_top = feature_importance.head(top_n)
colors = plt.cm.viridis(
    np.linspace(0.2, 0.9, top_n))

fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(fi_top["feature"][::-1],fi_top["importance"][::-1],color=colors[::-1])
ax.set_title("Top 20 Feature Importance (Gain)", fontsize=14)
ax.set_xlabel("Total Gain")
ax.set_ylabel("Feature")
plt.tight_layout()


# ÖNCE KAYDET
plt.savefig(
    "new_feature_importance_gain_top20_colored.png",
    dpi=300,
    bbox_inches="tight")
plt.show()
plt.close()


# =========================
# ROC Curve hesaplama
# =========================


# PR-AUC varken neden ROC da gösteriyorsunuz?
# PR-AUC dengesiz veri için ana metriktir,
# ROC–AUC ise modelin genel ayırt etme kabiliyetini göstermesi
# açısından destekleyici olarak sunulmuştur.

from sklearn.metrics import roc_curve, roc_auc_score

y_true = y_test
y_scores = y_prob  # predict_proba çıktısı ([:,1])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = roc_auc_score(y_true, y_scores)

# =========================
# ROC Curve çizimi
# =========================
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Fatal Accident Prediction")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

# =========================
# PNG olarak kaydet
# =========================
plt.savefig("roc_curve_final_model.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# =========================
# FN ANALIZI
# =========================
print("\n" + "="*90)
print("FN ANALİZİ (Kaçırılan fatal vakalar) - Sunum için debug")
print("="*90)

fn_mask = (y_test == 1) & (y_pred == 0)
fatal_mask = (y_test == 1)

fn_count = int(fn_mask.sum())
fatal_count = int(fatal_mask.sum())

print(f"Toplam test fatal: {fatal_count}")
print(f"Kaçırılan fatal (FN): {fn_count}")
print(f"Fatal kaçırma oranı: {fn_count / fatal_count:.4f}")

"""
Toplam test fatal: 2177
Kaçırılan fatal (FN): 200
Fatal kaçırma oranı: 0.0919
"""

X_fn = X_test.loc[fn_mask].copy()
X_fatal = X_test.loc[fatal_mask].copy()

fn_debug = X_fn.copy()
fn_debug["y_true"] = 1
fn_debug["y_prob"] = pd.Series(y_prob, index=X_test.index).loc[fn_mask]


fn_debug_sorted = fn_debug.sort_values("y_prob", ascending=False)

print("\nEn yüksek olasılıkla kaçırılan ilk 10 FN (sınırda kalanlar):")
print(fn_debug_sorted[["y_prob"]].head(10))

fn_debug_sorted.to_csv("false_negatives_debug.csv", index=True)
print("\nfalse_negatives_debug.csv kaydedildi")

"""
En yüksek olasılıkla kaçırılan ilk 10 FN (sınırda kalanlar):
        y_prob
912045   0.199
88606    0.199
254511   0.199
15732    0.199
826480   0.198
333871   0.198
335650   0.197
857167   0.197
788616   0.197
159169   0.197
"""
# =========================
# LIFT fonksiyonları
# =========================
def lift_table_binary(col_name, X_fn_, X_fatal_):
    a = (X_fn_[col_name].value_counts(normalize=True) * 100).rename("FN_%")
    b = (X_fatal_[col_name].value_counts(normalize=True) * 100).rename("AllFatal_%")
    out = pd.concat([a, b], axis=1).fillna(0)
    out["Lift(FN/AllFatal)"] = np.where(out["AllFatal_%"] > 0, out["FN_%"] / out["AllFatal_%"], np.nan)
    return out.sort_values("Lift(FN/AllFatal)", ascending=False)

def onehot_to_category(df_, prefix, reference_name="Reference/Other"):
    cols = [c for c in df_.columns if c.startswith(prefix)]
    if len(cols) == 0:
        raise ValueError(f"'{prefix}' ile başlayan kolon bulunamadı.")
    mat = df_[cols].values
    max_idx = mat.argmax(axis=1)
    max_val = mat.max(axis=1)
    cat = np.array([cols[i].replace(prefix, "") for i in max_idx], dtype=object)
    cat[max_val == 0] = reference_name
    return pd.Series(cat, index=df_.index)

def lift_table_categorical(cat_fn, cat_fatal):
    a = (cat_fn.value_counts(normalize=True) * 100).rename("FN_%")
    b = (cat_fatal.value_counts(normalize=True) * 100).rename("AllFatal_%")
    out = pd.concat([a, b], axis=1).fillna(0)
    out["Lift(FN/AllFatal)"] = np.where(out["AllFatal_%"] > 0, out["FN_%"] / out["AllFatal_%"], np.nan)
    return out.sort_values("Lift(FN/AllFatal)", ascending=False)

print("\n" + "="*90)
print("FN vs AllFatal LIFT TABLOLARI (Sunum için)")
print("="*90)

binary_cols = ["NEW_Is_Dark_1", "NEW_Is_Rural_1", "NEW_Is_Weekend_1", "NEW_Late_Night_1"]
for col in binary_cols:
    if col in X_test.columns:
        print("\n" + "-"*70)
        print(f"(1) {col}")
        print(lift_table_binary(col, X_fn, X_fatal))
    else:
        print(f"\n[SKIP] {col} yok")

if "Speed_limit" in X_test.columns:
    bins = [-1, 20, 30, 40, 50, 60, 70, 200]
    labels = ["<=20", "30", "40", "50", "60", "70", "70+"]

    fn_speed_bin = pd.cut(X_fn["Speed_limit"], bins=bins, labels=labels)
    fatal_speed_bin = pd.cut(X_fatal["Speed_limit"], bins=bins, labels=labels)

    a = (fn_speed_bin.value_counts(normalize=True) * 100).rename("FN_%")
    b = (fatal_speed_bin.value_counts(normalize=True) * 100).rename("AllFatal_%")

    out = pd.concat([a, b], axis=1).fillna(0)
    out["Lift(FN/AllFatal)"] = np.where(out["AllFatal_%"] > 0, out["FN_%"] / out["AllFatal_%"], np.nan)

    print("\n" + "-"*70)
    print("(2) Speed_limit (binlenmiş)")
    print(out.sort_values("Lift(FN/AllFatal)", ascending=False))
else:
    print("\n[SKIP] Speed_limit yok")

onehot_prefixes = [
    ("Road_Type_", "Road_Type"),
    ("Junction_Control_", "Junction_Control"),
    ("Road_Surface_Conditions_", "Road_Surface_Conditions"),
    ("Weather_Conditions_", "Weather_Conditions"),
    ("Light_Conditions_", "Light_Conditions"),
]

for prefix, name in onehot_prefixes:
    try:
        fn_cat = onehot_to_category(X_fn, prefix=prefix)
        fat_cat = onehot_to_category(X_fatal, prefix=prefix)

        print("\n" + "-"*70)
        print(f"(3) {name} (one-hot -> kategori)")
        print(lift_table_categorical(fn_cat, fat_cat).head(15))
    except Exception as e:
        print(f"\n[SKIP] {name} -> {e}")

"""
==========================================================================================
FN vs AllFatal LIFT TABLOLARI (Sunum için)
==========================================================================================

----------------------------------------------------------------------
(1) NEW_Is_Dark_1
                FN_%  AllFatal_%  Lift(FN/AllFatal)
NEW_Is_Dark_1                                      
False         70.000      58.199              1.203
True          30.000      41.801              0.718

----------------------------------------------------------------------
(1) NEW_Is_Rural_1
                 FN_%  AllFatal_%  Lift(FN/AllFatal)
NEW_Is_Rural_1                                      
False          79.500      36.886              2.155
True           20.500      63.114              0.325

----------------------------------------------------------------------
(1) NEW_Is_Weekend_1
                   FN_%  AllFatal_%  Lift(FN/AllFatal)
NEW_Is_Weekend_1                                      
False            81.000      66.651              1.215
True             19.000      33.349              0.570

----------------------------------------------------------------------
(1) NEW_Late_Night_1
                   FN_%  AllFatal_%  Lift(FN/AllFatal)
NEW_Late_Night_1                                      
False            98.000      87.644              1.118
True              2.000      12.356              0.162

----------------------------------------------------------------------
(2) Speed_limit (binlenmiş)
              FN_%  AllFatal_%  Lift(FN/AllFatal)
Speed_limit                                      
30          81.500      35.554              2.292
<=20         1.500       0.689              2.177
40           8.000       9.554              0.837
70           3.500      12.678              0.276
50           1.500       6.155              0.244
60           4.000      35.370              0.113
70+          0.000       0.000                NaN

----------------------------------------------------------------------
(3) Road_Type (one-hot -> kategori)
                     FN_%  AllFatal_%  Lift(FN/AllFatal)
Unknown             2.000       0.230              8.708
Roundabout         12.500       1.746              7.161
Slip_road           1.500       0.367              4.082
One_way_street      3.500       0.919              3.810
Single_carriageway 71.500      77.584              0.922
Reference/Other     9.000      19.155              0.470

----------------------------------------------------------------------
(3) Junction_Control (one-hot -> kategori)
                           FN_%  AllFatal_%  Lift(FN/AllFatal)
Automatic_traffic_signal 16.500       4.961              3.326
Stop_Sign                 1.500       0.505              2.969
Giveway_or_uncontrolled  64.500      31.465              2.050
Unknown                  17.500      63.023              0.278
Reference/Other           0.000       0.046              0.000

----------------------------------------------------------------------
(3) Road_Surface_Conditions (one-hot -> kategori)
                            FN_%  AllFatal_%  Lift(FN/AllFatal)
Snow                       2.000       0.413              4.838
Frost_Ice                  6.500       2.389              2.721
Flood__Over_3cm_of_water_  0.500       0.413              1.209
Reference/Other           65.500      67.294              0.973
Wet_Damp                  25.500      29.398              0.867
Unknown                    0.000       0.092              0.000

----------------------------------------------------------------------
(3) Weather_Conditions (one-hot -> kategori)
                             FN_%  AllFatal_%  Lift(FN/AllFatal)
Other                       5.000       1.654              3.024
Unknown                     2.500       1.470              1.701
Snowing_without_high_winds  0.500       0.413              1.209
Raining_without_high_winds 10.500       9.554              1.099
Fine_without_high_winds    78.500      82.912              0.947
Raining_with_high_winds     1.500       1.608              0.933
Reference/Other             1.500       1.608              0.933
Fog_or_mist                 0.000       0.781              0.000

----------------------------------------------------------------------
(3) Light_Conditions (one-hot -> kategori)
                                            FN_%  AllFatal_%  Lift(FN/AllFatal)
Darkness__Street_lights_present_but_unlit  2.000       0.965              2.073
Darkness__Street_lights_present_and_lit   26.000      21.084              1.233
Daylight__Street_light_present            70.000      58.199              1.203
Darkness__Street_lighting_unknown          0.500       1.011              0.495
Reference/Other                            1.500      18.741              0.080
"""
# =========================
# TOP 10 FN tablo
# =========================
top10_fn = fn_debug_sorted.head(10).copy()
show_cols = ["y_prob"]

for c in ["Speed_limit", "Hour", "NEW_Is_Dark_1", "NEW_Is_Rural_1", "NEW_Late_Night_1"]:
    if c in top10_fn.columns:
        show_cols.append(c)

print("\n" + "="*90)
print("TABLO: En sınırda kaçan 10 FN (p yüksek ama yine de kaçmış)")
print("="*90)

"""
==========================================================================================
TABLO: En sınırda kaçan 10 FN (p yüksek ama yine de kaçmış)
==========================================================================================
        y_prob  Speed_limit   Hour  NEW_Is_Dark_1  NEW_Is_Rural_1  NEW_Late_Night_1
912045   0.199           30 17.000           True           False             False
88606    0.199           30 19.000          False           False             False
254511   0.199           30 15.000          False           False             False
15732    0.199           30  8.000          False           False             False
826480   0.198           30 15.000          False           False             False
333871   0.198           30 23.000           True           False             False
335650   0.197           30 18.000          False           False             False
857167   0.197           30  7.000           True            True             False
788616   0.197           30 10.000          False           False             False
159169   0.197           30 15.000          False           False             False
"""

print(top10_fn[show_cols])
top10_fn[show_cols].to_csv("top10_borderline_FN_table.csv", index=True)
print("\ntop10_borderline_FN_table.csv kaydedildi")
# =========================
# SHAP ANALIZI
# =========================
print("\n" + "="*90)
print("SHAP ANALİZİ (Global + FN Local) - Sunumluk")
print("="*90)

# ---------------------------------------------------------
# Burada hangi modeli açıklamak istiyorsam onu seçiyorum
# ---------------------------------------------------------
# model = xgb_final
model = lgbm_final

# ---------------------------------------------------------
# 1) SHAP çok büyük veride ağır olur diye sample alıyorum
# ---------------------------------------------------------
RANDOM_STATE = 17

# background: TreeExplainer için referans set (küçük tutmak iyi)
bg_size = min(20000, len(X_train))
X_bg = X_train.sample(n=bg_size, random_state=RANDOM_STATE)

# explanation sample: global özet için
exp_size = min(10000, len(X_test))
X_exp = X_test.sample(n=exp_size, random_state=RANDOM_STATE)

print(f"Background size: {X_bg.shape}")
print(f"Explain sample size: {X_exp.shape}")

# Background size: (20000, 74)
# Explain sample size: (10000, 74)

# ---------------------------------------------------------
# 2) Explainer
# ---------------------------------------------------------
# Tree tabanlı modellerde en doğru/kolay: TreeExplainer
explainer = shap.TreeExplainer(model)

# Yeni SHAP API: explainer(X) -> Explanation objesi
shap_values = explainer(X_exp)

# ---------------------------------------------------------
# 3) GLOBAL: En etkili feature’lar (bar + beeswarm)
# ---------------------------------------------------------
plt.figure()
shap.plots.bar(shap_values, max_display=20, show=False)
plt.title("SHAP - Top 20 Feature (Global Bar)")
plt.tight_layout()
plt.savefig("shap_global_bar_top20.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title("SHAP - Global Summary (Beeswarm)")
plt.tight_layout()
plt.savefig("shap_global_beeswarm_top20.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("shap_global_bar_top20.png ve shap_global_beeswarm_top20.png kaydedildi")

# ---------------------------------------------------------
# 4) FN CASE’LERİ: “Sınırda kaçan” ilk 3 FN için waterfall
#    (fn_debug_sorted zaten üretilmişti)
# ---------------------------------------------------------
if "fn_debug_sorted" in globals():
    # sınırda kaçanlar = y_prob yüksek olan FN’ler
    fn_top_ids = fn_debug_sorted.head(3).index.tolist()
    print("\nFN Top (sınırda kaçan) indeksler:", fn_top_ids) # FN Top (sınırda kaçan) indeksler: [912045, 88606, 254511]

    for i, idx in enumerate(fn_top_ids, start=1):
        x_one = X_test.loc[[idx]]
        sv_one = explainer(x_one)

        p_val = float(pd.Series(y_prob, index=X_test.index).loc[idx])

        plt.figure()
        shap.plots.waterfall(sv_one[0], max_display=15, show=False)
        plt.title(f"FN Waterfall #{i} | index={idx} | p={p_val:.3f}")
        plt.tight_layout()
        plt.savefig(f"shap_fn_waterfall_{i}_idx_{idx}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    print("FN waterfall görselleri kaydedildi (shap_fn_waterfall_*.png)")
else:
    print("\n[SKIP] fn_debug_sorted yok. Önce FN debug bloğunu çalıştırınca burada local SHAP açılır.")

# ---------------------------------------------------------
# 5) FN’ler için “en çok etkileyen ilk 10 feature” tablosu
#    - Sunum için
# ---------------------------------------------------------
if "fn_debug_sorted" in globals():
    fn_top_ids = fn_debug_sorted.head(10).index.tolist()
    rows = []

    for idx in fn_top_ids:
        x_one = X_test.loc[[idx]]
        sv_one = explainer(x_one)[0]   # tek örnek
        # her feature için katkı
        contrib = pd.Series(sv_one.values, index=X_test.columns)

        top_pos = contrib.sort_values(ascending=False).head(5)   # fatal’a itenler
        top_neg = contrib.sort_values(ascending=True).head(5)    # non-fatal’a itenler

        rows.append({
            "index": idx,
            "y_prob": float(fn_debug_sorted.loc[idx, "y_prob"]) if "y_prob" in fn_debug_sorted.columns else np.nan,
            "fatal_itiyor_top5": ", ".join([f"{k}({v:.3f})" for k, v in top_pos.items()]),
            "nonfatal_itiyor_top5": ", ".join([f"{k}({v:.3f})" for k, v in top_neg.items()])
        })

    fn_shap_table = pd.DataFrame(rows)
    fn_shap_table.to_csv("fn_shap_top10_table.csv", index=False)
    print("\nfn_shap_top10_table.csv kaydedildi")

###################### XGBoost ILE MODELLEME ###################################################################

# ======================
# 1) y ve X hazırlanır
# ======================

# Bagimli ve Bagimsiz degiskenlerin ayarlanmasi
y = (df["Accident_Severity"] == 1).astype(int)
X = df.drop(columns=["Accident_Severity"] + drop_liste)

# X'in uygun formata getirilmesi (kolon adlarını temizle)
def clean_feature_names(df_):
    df_ = df_.copy()
    df_.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df_.columns]
    return df_

X = clean_feature_names(X)

print("y shape:", y.shape)
print("X shape:", X.shape)

# Imbalance: scale_pos_weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print("scale_pos_weight:", scale_pos_weight)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

# ======================================
# 2) Base XGBoost Model (CV ile ölçüm)
# ======================================

xgb_model = XGBClassifier(
    objective="binary:logistic",
    random_state=17,
    scale_pos_weight=scale_pos_weight,   # kritik
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="hist",
    n_jobs=-1,
    eval_metric="auc"
)

cv_results = cross_validate(
    xgb_model,
    X, y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    n_jobs=-1
)

print("\n--- BASE CV RESULTS ---")
print("CV Accuracy :", cv_results["test_accuracy"].mean())
print("CV Precision:", cv_results["test_precision"].mean())
print("CV Recall   :", cv_results["test_recall"].mean())
print("CV F1       :", cv_results["test_f1"].mean())
print("CV ROC-AUC  :", cv_results["test_roc_auc"].mean())

# ==========================================================
# 3) (Opsiyonel) GridSearch ile en iyi modeli bulmak istersen
#    (Çok uzun sürebilir, istersen aç)
# ==========================================================

DO_GRIDSEARCH = True  # True yaparsan çalışır ama uzun sürer

if DO_GRIDSEARCH:
    xgb_params = {
        "max_depth": [6, 8],
        "min_child_weight": [1, 3],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "learning_rate": [0.03, 0.05],
        "n_estimators": [500, 800],
        "reg_lambda": [1.0, 2.0]
    }

    grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=xgb_params,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=2
    ).fit(X, y)

    print("\nBest params:", grid.best_params_)
    # GridSearch sonrası final model:
    xgb_final = grid.best_estimator_
else:
    # GridSearch yapılmazsa final model = base model
    xgb_final = xgb_model

# ======================================
# 4) Train/Test + Threshold Tuning
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=17
)

#
xgb_final.fit(X_train, y_train)

y_prob = xgb_final.predict_proba(X_test)[:, 1]

print("\n--- THRESHOLD TUNING ---")
for t in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    y_pred = (y_prob >= t).astype(int)
    print(
        f"t={t:.2f} | "
        f"Precision={precision_score(y_test, y_pred):.3f} | "
        f"Recall={recall_score(y_test, y_pred):.3f} | "
        f"F1={f1_score(y_test, y_pred):.3f}"
    )

""" 
( vehicles silmeden önce)
--- THRESHOLD TUNING ---
t=0.03 | Precision=0.014 | Recall=0.953 | F1=0.028
t=0.05 | Precision=0.015 | Recall=0.917 | F1=0.030
t=0.08 | Precision=0.017 | Recall=0.886 | F1=0.034
t=0.10 | Precision=0.019 | Recall=0.861 | F1=0.036
t=0.12 | Precision=0.020 | Recall=0.835 | F1=0.039
t=0.15 | Precision=0.021 | Recall=0.793 | F1=0.042
t=0.20 | Precision=0.024 | Recall=0.736 | F1=0.047
---------------------------------------------------
(vehicles sildikten sonra)
--- THRESHOLD TUNING ---
t=0.03 | Precision=0.013 | Recall=0.952 | F1=0.026
t=0.05 | Precision=0.014 | Recall=0.926 | F1=0.028
t=0.08 | Precision=0.016 | Recall=0.882 | F1=0.031
t=0.10 | Precision=0.017 | Recall=0.858 | F1=0.033
t=0.12 | Precision=0.018 | Recall=0.828 | F1=0.034
t=0.15 | Precision=0.019 | Recall=0.792 | F1=0.037
t=0.20 | Precision=0.022 | Recall=0.723 | F1=0.042


# Threshold seçimi (Safety-critical):
# Öncelik: ölümcül (fatal) kazaları kaçırmamak => Recall yüksek tutulmalı.
# Ancak çok düşük threshold (0.05 gibi) aşırı yanlış alarm üretiyor (Precision çok düşüyor).
# Bu yüzden "Recall >= 0.82" şartını koyup, bu şartı sağlayan threshold'ler içinde
# precision'ı daha iyi olan denge noktasını seçtik.
# Sonuç: t=0.12 -> yüksek recall + daha yönetilebilir yanlış alarm dengesi.
"""
# ======================================
# 5) Confusion Matrix (seçilen threshold)
# ======================================

t = 0.12
y_pred = (y_prob >= t).astype(int)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix raw:\n", cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Severe (0)", "Severe (1)"]
)

plt.figure(figsize=(6, 5))
disp.plot(values_format="d")
plt.title(f"XGBoost Confusion Matrix (Threshold = {t})")
plt.show()

"""
Confusion matrix raw:
 [[ 83497 101144]
 [   374   1803]]
"""
# ======================================
# 6) Feature Importance (Gain)
# ======================================

booster = xgb_final.get_booster()
score_gain = booster.get_score(importance_type="gain")

fi = pd.DataFrame({
    "feature": X.columns,
    "importance": [score_gain.get(f, 0.0) for f in X.columns]
}).sort_values("importance", ascending=False)

print("\nTop 20 Feature Importance (gain):")
print(fi.head(20))

fi.head(20).set_index("feature")["importance"].plot(kind="barh", figsize=(8, 6))
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importance (Gain) - XGBoost")
plt.show()
"""
Top 20 Feature Importance (gain):
                                            feature  importance
4                                       Speed_limit     283.954
10                             Road_Type_Roundabout     258.764
66                                    NEW_Is_Dark_1     229.103
17                         Junction_Control_Unknown     208.984
65                                   NEW_Is_Rural_1     142.148
73                        NEW_Speed_Road_Mismatch_1     136.471
28  Light_Conditions_Daylight__Street_light_present     133.039
70                         NEW_No_Street_Lighting_1     132.267
72                             NEW_Ped_Risk_Proxy_1     131.009
64                                 NEW_Late_Night_1     125.635
39                     Road_Surface_Conditions_Snow     119.179
63                                 NEW_Is_Weekend_1     116.664
11                     Road_Type_Single_carriageway     113.688
38                Road_Surface_Conditions_Frost_Ice     111.828
8                                              Hour     108.415
12                              Road_Type_Slip_road     107.566
2                                    1st_Road_Class     107.023
29       Weather_Conditions_Fine_without_high_winds     104.488
68                           NEW_Bad_Road_Surface_1     101.449
14        Junction_Control_Automatic_traffic_signal     101.235
"""
# ======================================
# 7) Top-35 feature ile tekrar CV
# ======================================

top_features = fi.head(35)["feature"].tolist()
X_top = X[top_features]

cv_results_top = cross_validate(
    xgb_final,
    X_top, y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    n_jobs=-1
)

print("\n--- TOP35 CV RESULTS ---")
print("TOP35 CV Accuracy :", cv_results_top["test_accuracy"].mean())
print("TOP35 CV Precision:", cv_results_top["test_precision"].mean())
print("TOP35 CV Recall   :", cv_results_top["test_recall"].mean())
print("TOP35 CV F1       :", cv_results_top["test_f1"].mean())
print("TOP35 CV ROC-AUC  :", cv_results_top["test_roc_auc"].mean())

"""
--- TOP35 CV RESULTS ---
TOP35 CV Accuracy : 0.8517453899040849
TOP35 CV Precision: 0.03787995200588481
TOP35 CV Recall   : 0.48029912858910234
TOP35 CV F1       : 0.07022159989617723
TOP35 CV ROC-AUC  : 0.7392168259376412
"""

# ===========================================================================
# 8) PR CURVE + CONFUSION MATRIX (Threshold seçimimii desteklemek için)
# ===========================================================================
from sklearn.metrics import precision_recall_curve, average_precision_score
# Bu problemde pozitif sınıf (ölümcül / severe) çok az.
# O yüzden ROC-AUC tek başına yeterli olmuyor.
# Ben burada özellikle Precision-Recall eğrisine bakıyorum çünkü:
# - "Ölümcül kazaları yakalayabiliyor muyum?" (Recall) çok kritik
# - "Ne kadar yanlış alarm veriyorum?" (Precision) da önemli ama ikinci planda
# PR curve bu dengeyi daha net gösteriyor.

# Benim seçtiğim threshold (t)
# precision zaten düşük olacak ama bu veri dengesiz olduğu için beklediğim bir durum.
t = 0.12

# ------------------------------------------------------------
# Precision-Recall curve çıkarıyorum
# ------------------------------------------------------------
# y_prob: modelin "1 sınıfı" için verdiği olasılık (predict_proba[:,1])
prec, rec, thr = precision_recall_curve(y_test, y_prob)

# PR-AUC (Average Precision) -> PR curve altında kalan alan gibi düşünebilirsin
ap = average_precision_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(rec, prec)
plt.xlabel("Recall (Ölümcül kazaları yakalama oranı)")
plt.ylabel("Precision (Alarm verdiğim kazaların ne kadarı gerçekten ölümcül?)")
plt.title(f"Precision-Recall Curve (AP / PR-AUC = {ap:.4f})")

# ------------------------------------------------------------
# Seçtiğim threshold noktasını PR curve üzerinde işaretliyorum
# ------------------------------------------------------------
# precision_recall_curve fonksiyonunda threshold array'i 1 eleman kısa geliyor,
# o yüzden en yakın threshold indexini bulup işaretliyorum.
idx = np.argmin(np.abs(thr - t))

p_t = prec[idx]
r_t = rec[idx]

plt.scatter(r_t, p_t, s=80)
plt.annotate(
    f"t={t:.2f}\nPrecision={p_t:.3f}, Recall={r_t:.3f}",
    (r_t, p_t),
    textcoords="offset points",
    xytext=(10, -10)
)

plt.grid(True, alpha=0.2)
plt.tight_layout()

# Sunumda direkt kullanayım diye görseli kaydediyorum:
plt.savefig("pr_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# Confusion Matrix (t=0.08 ile) + Classification Report
# ------------------------------------------------------------
# Threshold uygulayıp 0/1 tahmine çeviriyorum
y_pred = (y_prob >= t).astype(int)

print("\n--- Classification Report (t=0.08) ---")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix raw:\n", cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Severe (0)", "Severe (1)"]
)

plt.figure(figsize=(6, 5))
disp.plot(values_format="d")
plt.title(f"Confusion Matrix (Threshold = {t})")
plt.tight_layout()

# Sunum için bunu da kaydediyorum:
plt.savefig("confusion_matrix_t012.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# 9) HATA ANALİZİ (FN): Modelin kaçırdığı fatal kazalar
# =========================================================
# Benim için en kritik şey şu:
# "Model fatal kazaları kaçırmasın."
# O yüzden özellikle FN (False Negative) örneklerine bakıyorum:
#   - Gerçek fatal (1) ama model non-fatal (0) demiş -> modelin kaçırdığı durumlar
#
# Bu analiz sunumda da çok güçlü oluyor:
# "Model nerelerde zorlanıyor / hangi koşullarda fatal’i kaçırıyor?"

t = 0.12
y_pred = (y_prob >= t).astype(int)

# X_test DataFrame değilse DataFrame'e çeviriyorum (genelde zaten DataFrame oluyor)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test, columns=X.columns)

# -----------------------------
# FN / TP / Tüm Fatal maskeleri
# -----------------------------
fn_mask = (y_test == 1) & (y_pred == 0)   # kaçırılan fatal
tp_mask = (y_test == 1) & (y_pred == 1)   # yakalanan fatal
pos_mask = (y_test == 1)                  # tüm fatal (gerçek=1)

X_fn  = X_test.loc[fn_mask].copy()
X_pos = X_test.loc[pos_mask].copy()

print("Toplam test fatal sayısı:", int(pos_mask.sum()))
print("Kaçırılan fatal (FN):", int(fn_mask.sum()))
print("Yakalanan fatal (TP):", int(tp_mask.sum()))
print("Fatal kaçırma oranı (FN / tüm fatal):", round(fn_mask.sum() / pos_mask.sum(), 4))

"""
Toplam test fatal sayısı: 2177
Kaçırılan fatal (FN): 374
Yakalanan fatal (TP): 1803
Fatal kaçırma oranı (FN / tüm fatal): 0.1718
"""
# =========================================================
# One-hot kolonlardan kategori geri çıkarma (Road_Type vs.)
# =========================================================
# Bazı kolonlar one-hot olduğu için (Road_Type_..., Weather_..., vb.)
# her satırda "hangi kategori aktif?" sorusunu burada çözüyorum.
# Eğer satırda hepsi 0 ise (drop_first gibi durumlarda) "Reference/Other" yazıyorum.

def dominant_onehot_category(df, prefix, reference_label="Reference/Other"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if len(cols) == 0:
        return None

    sub = df[cols]
    idx = sub.values.argmax(axis=1)     # en yüksek (genelde 1 olan) kolon
    maxv = sub.values.max(axis=1)       # hepsi 0 mı kontrolü

    chosen = np.array(cols, dtype=object)[idx]
    chosen = pd.Series(chosen, index=df.index)

    chosen[maxv == 0] = reference_label
    chosen = chosen.str.replace(prefix, "", regex=False)
    return chosen


# =========================================================
# FN vs Tüm Fatal karşılaştırması (oran + lift)
# =========================================================
# Ben burada şunu görmek istiyorum:
# "Modelin kaçırdığı fatal'lar hangi koşullarda daha sık?"
# Bunun için FN içindeki oranı, tüm fatal içindeki oranla kıyaslıyorum.
# Lift > 1 çıkarsa: o kategori FN’de daha baskın -> model burada daha çok kaçırıyor demek.

def compare_share(fn_series, pos_series, top_k=8, title=""):
    fn_rate = fn_series.value_counts(normalize=True) * 100
    pos_rate = pos_series.value_counts(normalize=True) * 100

    out = pd.DataFrame({
        "FN_%": fn_rate,
        "AllFatal_%": pos_rate
    }).fillna(0)

    out["Lift(FN/AllFatal)"] = (out["FN_%"] / out["AllFatal_%"].replace(0, np.nan)).round(2)
    out = out.sort_values("Lift(FN/AllFatal)", ascending=False)

    print("\n" + "="*70)
    print(title)
    print(out.head(top_k).round(2))
    return out


# =========================================================
# Sorulacak başlıklar: gece mi / kırsal mı / hız / yol tipi / kavşak
# =========================================================

# (1) Gece mi?
if "NEW_Is_Dark_1" in X_test.columns:
    compare_share(
        X_fn["NEW_Is_Dark_1"], X_pos["NEW_Is_Dark_1"],
        title="(1) FN'ler karanlık mı? (NEW_Is_Dark_1)"
    )

# (2) Kırsal mı?
if "NEW_Is_Rural_1" in X_test.columns:
    compare_share(
        X_fn["NEW_Is_Rural_1"], X_pos["NEW_Is_Rural_1"],
        title="(2) FN'ler kırsal mı? (NEW_Is_Rural_1)"
    )

# (3) Hız limiti kaç?
# UK’de genelde 30/40/50/60/70 çok anlamlı, o yüzden bin yapıyorum.
if "Speed_limit" in X_test.columns:
    bins = [0, 20, 30, 40, 50, 60, 70, 200]
    labels = ["<=20", "30", "40", "50", "60", "70", "70+"]
    fn_speed_bin  = pd.cut(X_fn["Speed_limit"], bins=bins, labels=labels, include_lowest=True, right=True)
    pos_speed_bin = pd.cut(X_pos["Speed_limit"], bins=bins, labels=labels, include_lowest=True, right=True)

    compare_share(
        fn_speed_bin, pos_speed_bin,
        title="(3) FN'lerde Speed_limit dağılımı (binlenmiş)"
    )

# (4) Yol tipi (Road_Type one-hot -> kategori)
road_fn  = dominant_onehot_category(X_fn,  "Road_Type_")
road_pos = dominant_onehot_category(X_pos, "Road_Type_")
if road_fn is not None:
    compare_share(
        road_fn, road_pos,
        title="(4) FN'lerde Road_Type (one-hot -> kategori)"
    )

# (5) Kavşak kontrolü (Junction_Control one-hot -> kategori)
junc_fn  = dominant_onehot_category(X_fn,  "Junction_Control_")
junc_pos = dominant_onehot_category(X_pos, "Junction_Control_")
if junc_fn is not None:
    compare_share(
        junc_fn, junc_pos,
        title="(5) FN'lerde Junction_Control (one-hot -> kategori)"
    )

# (6) Yol yüzeyi / hava durumu
surf_fn  = dominant_onehot_category(X_fn,  "Road_Surface_Conditions_")
surf_pos = dominant_onehot_category(X_pos, "Road_Surface_Conditions_")
if surf_fn is not None:
    compare_share(surf_fn, surf_pos, title="(Bonus) FN'lerde Road_Surface_Conditions")

w_fn  = dominant_onehot_category(X_fn,  "Weather_Conditions_")
w_pos = dominant_onehot_category(X_pos, "Weather_Conditions_")
if w_fn is not None:
    compare_share(w_fn, w_pos, title="(Bonus) FN'lerde Weather_Conditions")

"""
======================================================================
(1) FN'ler karanlık mı? (NEW_Is_Dark_1)
                FN_%  AllFatal_%  Lift(FN/AllFatal)
NEW_Is_Dark_1                                      
False         60.160      58.200              1.030
True          39.840      41.800              0.950

======================================================================
(2) FN'ler kırsal mı? (NEW_Is_Rural_1)
                 FN_%  AllFatal_%  Lift(FN/AllFatal)
NEW_Is_Rural_1                                      
False          70.320      36.890              1.910
True           29.680      63.110              0.470

======================================================================
(3) FN'lerde Speed_limit dağılımı (binlenmiş)
              FN_%  AllFatal_%  Lift(FN/AllFatal)
Speed_limit                                      
<=20         2.140       0.690              3.100
30          68.980      35.550              1.940
40           8.820       9.550              0.920
70           5.880      12.680              0.460
50           2.410       6.160              0.390
60          11.760      35.370              0.330
70+          0.000       0.000                NaN

======================================================================
(4) FN'lerde Road_Type (one-hot -> kategori)
                     FN_%  AllFatal_%  Lift(FN/AllFatal)
Unknown             1.070       0.230              4.660
Roundabout          7.750       1.750              4.440
One_way_street      3.480       0.920              3.780
Slip_road           0.800       0.370              2.180
Single_carriageway 72.460      77.580              0.930
Reference/Other    14.440      19.150              0.750

======================================================================
(5) FN'lerde Junction_Control (one-hot -> kategori)
                           FN_%  AllFatal_%  Lift(FN/AllFatal)
Reference/Other           0.270       0.050              5.820
Stop_Sign                 1.870       0.510              3.700
Automatic_traffic_signal 17.380       4.960              3.500
Giveway_or_uncontrolled  53.210      31.470              1.690
Unknown                  27.270      63.020              0.430

======================================================================
(Bonus) FN'lerde Road_Surface_Conditions
                            FN_%  AllFatal_%  Lift(FN/AllFatal)
Unknown                    0.530       0.090              5.820
Snow                       1.070       0.410              2.590
Frost_Ice                  5.610       2.390              2.350
Flood__Over_3cm_of_water_  0.800       0.410              1.940
Wet_Damp                  31.820      29.400              1.080
Reference/Other           60.160      67.290              0.890

======================================================================
(Bonus) FN'lerde Weather_Conditions
                             FN_%  AllFatal_%  Lift(FN/AllFatal)
Other                       3.740       1.650              2.260
Unknown                     2.940       1.470              2.000
Snowing_without_high_winds  0.800       0.410              1.940
Raining_with_high_winds     2.410       1.610              1.500
Raining_without_high_winds 14.170       9.550              1.480
Fog_or_mist                 0.800       0.780              1.030
Fine_without_high_winds    73.800      82.910              0.890
Reference/Other             1.340       1.610              0.830

"""
# =========================================================
# FN kayıtlarını dışa aktarma (tek tek incelemek için)
# =========================================================
# Sunumdan önce:
# Kaçırılan fatal örneklerine bak
# Böylece gerçekten modelin zorlandığı gerçek senaryoları yakalamakta zorlanıyor

fn_debug = X_fn.copy()
fn_debug["y_true"] = 1
fn_debug["y_prob"] = y_prob[fn_mask]   # modelin verdiği olasılık

# En "sınırda" kaçırılanları görmek için (y_prob yüksek ama yine de 0 kalmış)
fn_debug_sorted = fn_debug.sort_values("y_prob", ascending=False)

print("\nEn yüksek olasılıkla kaçırılan ilk 10 FN örneği:")
print(fn_debug_sorted[["y_prob"]].head(10))

fn_debug_sorted.to_csv("false_negatives_debug.csv", index=True)
print("\nfalse_negatives_debug.csv kaydedildi ")

"""
En yüksek olasılıkla kaçırılan ilk 10 FN örneği:
        y_prob
88606    0.120
568886   0.120
46105    0.119
692911   0.118
897620   0.118
385984   0.117
757070   0.117
389947   0.117
159247   0.116
665016   0.116

Çıkarımlar:
FN’lerde 30 hız limiti şişkin (lift yüksek)

Giveway / uncontrolled junction control FN tarafında fazla

Roundabout / Unknown road type bazı FN’lerde aşırı lift veriyor (modelin zorlandığı senaryolar)

“Dark mı?” dağılımı çok dramatik değil


"""
# =========================
# SHAP'e geçiş (Top 10 FN)
# =========================
import shap
# Mantık:
# - top10_idx: "kaçırılan fatal" örnekler içinden y_prob'u en yüksek olan ilk 10 kayıt
# - SHAP: modelin bu örneklerde hangi feature'lara bakıp nasıl karar verdiğini açıklıyor

top10_idx = fn_debug_sorted.head(10).index

# Eğer indexler X_test ile uyumluysa en temizi buradan çekmek:
X_top10_fn = X_test.loc[top10_idx].copy()

# SHAP TreeExplainer (XGBoost için çok uygun)
explainer = shap.TreeExplainer(xgb_final)

# SHAP değerleri (her satır için her feature'ın etkisi)
shap_values = explainer.shap_values(X_top10_fn)
base_value = explainer.expected_value

# -------------------------------
# 4) Sunum için tablo çıkarıyorum
# -------------------------------
# Ben burada her FN örnek için:
# "bu örnekte modeli en çok etkileyen ilk 8 feature ne?" onu çıkartıyorum.
# Böylece slayta koyabileceğim net bir tablo oluyor.

rows = []
for i, row_id in enumerate(X_top10_fn.index):
    vals = shap_values[i]
    order = np.argsort(np.abs(vals))[::-1][:8]  # en etkili 8 feature

    for k in order:
        rows.append({
            "case_id": row_id,
            "y_prob": float(fn_debug_sorted.loc[row_id, "y_prob"]),
            "feature": X_top10_fn.columns[k],
            "value": X_top10_fn.iloc[i, k],
            "shap_effect": float(vals[k])   # (+) fatal'e iter, (-) fatal'den uzaklaştırır
        })

shap_table = pd.DataFrame(rows)

# daha okunur olsun diye: her case_id içinde SHAP etkisine göre sıralıyorum
shap_table["abs_effect"] = shap_table["shap_effect"].abs()
shap_table = shap_table.sort_values(["case_id", "abs_effect"], ascending=[True, False]).drop(columns=["abs_effect"])

shap_table.to_csv("top10_fn_shap_table.csv", index=False)
print("\ntop10_fn_shap_table.csv kaydedildi")

# -------------------------------
# 5) Sunum için 1-2 tane görsel (waterfall) kaydediyorum
# -------------------------------
# Waterfall plot: tek bir örnekte model kararını adım adım gösteriyor.
# Ben genelde sunumda 1-2 tane koyuyorum, fazla koyunca kalabalık oluyor.

for i in range(min(2, len(X_top10_fn))):
    exp = shap.Explanation(
        values=shap_values[i],
        base_values=base_value,
        data=X_top10_fn.iloc[i],
        feature_names=X_top10_fn.columns
    )

    plt.figure()
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title(f"FN Case {X_top10_fn.index[i]} | p={fn_debug_sorted.loc[X_top10_fn.index[i], 'y_prob']:.3f} (ama yine de kaçmış)")
    plt.tight_layout()
    plt.savefig(f"fn_waterfall_{i+1}.png", dpi=220)
    plt.close()

print("fn_waterfall_1.png ve fn_waterfall_2.png kaydedildi")

"""
p = y_prob
p >= t : fatal(1)
p < t : fatal değil ((0)

Model fatalı nerede kaçırıyor?
-> FN’ler kırsal değil, daha çok “urban” tarafta
NEW_Is_Rural_1=False için lift 1.91
Yani fatal vakaların genelinde kırsal ağırlık yüksekken, kaçırdıkları görece şehir içi.

-> Speed_limit 30 FN’lerde aşırı baskın
Speed_limit=30 lift 1.94
Bu çok önemli: model fatal’i 30 mph ortamında daha çok kaçırıyor.

-> Road_Type: Roundabout / One-way / Unknown kaçırılanlarda yüksek lift
Roundabout lift 4.44
One_way_street lift 3.78
Unknown lift 4.66
Yani model fatali özellikle roundabout gibi “karmaşık geometrilerde” daha çok kaçırıyor.

-> Junction_Control: traffic signal ve giveway/uncontrolled FN’de yüksek
Automatic_traffic_signal lift 3.50
Giveway_or_uncontrolled lift 1.69
Stop_Sign lift 3.70
Bu da “kavşak kontrol tiplerinde” modelin zorlandığını gösteriyor.

-> Zemin/hava bonusları:
Frost_Ice lift 2.35
Raining_without_high_winds lift 1.48
Model bazı “kötü koşullarda” fatal’i daha çok kaçırıyor.

ÖZETLE : Modelin kaçırdığı fatallar => urban + 30 mph + roundabout/traffic signal kombinasyonlarına daha yakın.

------- En yüksek olasılıkla kaçırılan FN örnekleri ------
y_prob değerleri 0.116–0.120 bandında
Model bu kazalar için “fatal olma ihtimali yüksek” demiş, ama 0.12 barajını çok az geçemediği için negatifte kalmış (FN)
Bunlar borderline (near-miss) örnekleridir

0.120 gözüktüğü halde nasıl FN oldu?
1. Ekranda 0.120 görünür ama gerçek değer 0.11996 gibi olabilir → 0.12’nin altında kalır ve FN olur
2. Bazı kodlarda y_prob > t kullanılır (>= değil). O zaman tam 0.120 olanlar da negatifte kalabilir.

ÖZETLE: Modelin kaçırdığı fatal vakaların bir kısmı eşik değerine çok yakın. Bu, modelin bu örneklerde tamamen kör olmadığı; ancak kararın eşiğe takıldığı anlamına gelir.

------------ Waterfall Grafikler Hakkında -----------------
Görsel 1 – FN Case 88606 (p=0.120, f(x) = -1.996):
Bu FN örneğinde model, 30 mph + yol sınıfı/numarası + saat sinyallerini ‘daha düşük fatal risk’ gibi öğrenmiş. Fatal gerçekleşse de, bu güçlü mavi etkiler yüzünden skor eşikte kalmış

Görsel 2 – FN Case 568886 (p=0.120):
Bu örnek ‘modelin gözünde’ tipik düşük risk profili (30 mph + gündüz + aydınlık). Fatal olmasına rağmen elimizdeki feature’lar bu vakayı fatal olarak ayırmaya yetmemiş; model güçlü bir ‘fatal sinyali’ görememiş.

"""

#######    GIRDI KULLANILIP FATAL OLUP OLMAMA DURUMUNUN SORGULANMASI  ############################
##################################################################################################
new_observation=[]

def predict_fatal_risk(model, X_row, threshold=0.15):
    prob = model.predict_proba(X_row)[:, 1][0]
    pred = int(prob >= threshold)

    return {
        "fatal_probability": prob,
        "prediction": pred,
        "risk_level": "High" if pred == 1 else "Low"
    }
result = predict_fatal_risk(lgbm_final, new_observation)

print(result)


#Girdi: Kaza koşulları

#Çıktı: Ölümcül olma olasılığı

#Karar: Threshold tabanlı risk sınıflaması

# Model Çıktısı
#
# Fatal Olasılığı: 0.27
#
# Threshold: 0.20

#Tahmin:  Yüksek Risk (Fatal olabilir)



