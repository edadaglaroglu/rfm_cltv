###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_csv('Flo_rfm_case/flo_data_20k.csv')
df = df_.copy()

# 2. data set
# a. İlk 10 gözlem,
df.head(10)
# b. Değişken isimleri,
df.columns
# c. Boyut,
df.shape
# d. Betimsel istatistik,
df.describe().T
# e. Boş değer,
df.isnull().sum()
# f. Değişken tipleri, incelemesi yapın
df.info()
# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["total_number_of_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()
# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
# df[selected_col].astype(datetime64[ns])
selected_col = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
for col in selected_col:
    df[col] = pd.to_datetime(df[col])

## Farklı çözüm
for i in df.columns:
    if "date" in i:
        df[i] = pd.to_datetime(df[i])

# df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)

# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız.

df_2 = df.groupby("order_channel").agg({"master_id": "count",
                                        "total_number_of_purchases": "sum",
                                        "total_price": "sum"})

df_2

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.sort_values(by="total_price", ascending=False).head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.sort_values(by="total_number_of_purchases", ascending=False).head(10)


# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def preparation(dataframe):
    # dataframe.head(10)
    # dataframe.columns
    # dataframe.shape
    # dataframe.describe().T
    # dataframe.isnull().sum()
    # dataframe.info()

    # adding new columns
    dataframe["total_number_of_purchases"] = dataframe["order_num_total_ever_online"] + dataframe[
        "order_num_total_ever_offline"]
    dataframe["total_price"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]

    # objject to datetime

    selected_col = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for col in selected_col:
        df[col] = pd.to_datetime(df[col])

    # dataframe["last_order_date"] = dataframe["last_order_date"].apply(pd.to_datetime)
    # dataframe["first_order_date"] = dataframe["first_order_date"].apply(pd.to_datetime)
    # dataframe["last_order_date_online"] = dataframe["last_order_date_online"].apply(pd.to_datetime)
    # dataframe["last_order_date_offline"] = dataframe["last_order_date_offline"].apply(pd.to_datetime)

    return dataframe


df = df_.copy()
df = preparation(df)
df.head()

###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################
from datetime import datetime, timedelta
import datetime as dt

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi

df["last_order_date"].max()  # bu müşterinin yaptıgı son alılveriş tarihini verecek


today_date = dt.datetime(2021, 6, 1)  # tarih giriyorum bunu zaman değişkeni cinsinde oluştur

# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
# df["master_id"].nunique()
# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.

rfm = df.groupby("master_id").agg({'last_order_date': lambda x: (today_date - x.max()).days,
                                   "total_number_of_purchases": lambda y: y,
                                   "total_price": lambda z: z})
rfm.shape
rfm.head()
# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz
rfm.columns = ["recency", "frequency", "monetary"]

# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe

###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm.head()
# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
rfm.tail()
###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'}

rfm['segment'] = rfm["RF_SCORE"].replace(seg_map, regex=True)

rfm.head()
###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean"})

# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.

new_df = df[df["interested_in_categories_12"].apply(lambda x: "KADIN" in x)]
new_df.head()
bayan_ındex = new_df["master_id"]

rfm.reset_index(inplace=True)
rfm_cust = rfm.loc[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers"), "master_id"]
ortak_id_diger = rfm_cust[rfm_cust.isin(bayan_ındex)].tolist()


new_2 = pd.DataFrame()
new_2["customer_id"] = ortak_id_diger
new_2.to_csv("yeni_marka_hedef_müşteri_id.cvs", index=False)

## Farklı bir çözüm
_segment = rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")]
_sex = df[(df["interested_in_categories_12"]).str.contains("KADIN")]
_case_A = pd.merge(_segment, _sex[["interested_in_categories_12", "master_id"]], on=["master_id"])
_case_A[["master_id"]].to_csv("case_A.csv", index=False)

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.


new_df_2 = df[df["interested_in_categories_12"].apply(lambda x: (("ERKEK") or ("COCUK")) in x)]
sleected_category = new_df_2["master_id"]


rfm_selected = df.loc[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "new_customers"), "master_id"]

common_id = rfm_selected[rfm_selected.isin(sleected_category).to_list()]

new_3 = pd.DataFrame()
new_3["customer_id"] = common_id

new_3.to_csv("indirim_hedef_müşteri_ids.csv")

# isin [""]
# isin()
# isin() kullanarak da segmentleri seçebiliriz
x=rfm[rfm["segment"].isin(["hibernating","cant_loose","new_customers"])]
x.head()

# rfm[rfm[‘master_id'].isin(rfm[rfm['segment'].isin(['cant_loose', 'about_to_sleep', 'new_customers'])]['customer_id'])]

### Farklı çözüm
segment = rfm[(rfm["segment"] == "hibernating") | (rfm["segment"] == "new_custcant_loosemers") | (
            rfm["segment"] == "new_custmers")]
sex = df[((df["interested_in_categories_12"]).str.contains("ERKEK")) & (
    (df["interested_in_categories_12"]).str.contains("COCUK"))]
case_B = pd.merge(segment, sex[["interested_in_categories_12", "master_id"]], on=["master_id"])
case_B[["master_id"]].to_csv("case_B.csv", index=False)
