##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


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
# GÖREV 1: Veriyi Hazırlama
###############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv('FLOCLTVPrediction/flo_data_20k.csv')
df = df_.copy()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.
plt.boxplot(df["order_num_total_ever_online"])
plt.show()

num_col= [col for col in df.columns if df[col].dtypes != "O"]
for col in num_col:
    plt.boxplot(df[col])
    plt.title(col)
    plt.show()
##########

df.describe().T

for col in df.columns[df.columns.str.contains('total')]:
    replace_with_thresholds(df,col)


replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["total_number_of_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
selected_col=["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
for col in selected_col:
    df[col]= pd.to_datetime(df[col])


###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################
df.head()
import datetime as dt
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max()

today_date=dt.datetime(2021,6,1)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

# recency:müşteri order.max- order.min
# t: analiz_tarihi- müşterinin_ilk alışveriş tarihi
# fquency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç # burda ortalama diyor  toplan kazanç bölü frequency yapıcaz

df["recency_cltv_weekly"]=(df["last_order_date"]- df["first_order_date"]).dt.days

cltv_df=df.groupby("master_id").agg({"recency_cltv_weekly": lambda x: x,
                            "first_order_date": lambda x: (today_date- x.min()).days,
                             "total_number_of_purchases": lambda y: y.astype(int),
                             "total_price": lambda z: z})

cltv_df.head()

cltv_df.columns=["recency_cltv_weekly","T_weekly","frequency","monetary_cltv_avg"]

# satın alma başına ortalama dediği için monetary i frequency e bölecegiz
cltv_df["monetary_cltv_avg"] = cltv_df["monetary_cltv_avg"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]  # neden 1

cltv_df["recency_cltv_weekly"] = cltv_df["recency_cltv_weekly"] / 7  # # recency değeri haftalık oldugu için biz burda geçen gün sayısını bulduk 7 ye bölmemiz lazım ,haftalık cinsten kaç haftadır kendi içinde alışveriş yapmadıkalrı

cltv_df["T_weekly"] = cltv_df["T_weekly"] / 7

cltv_df.describe().T

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################
# expected frequency
# 1. BG/NBD modelini kurunuz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])


# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.


cltv_df["exp_sales_3_month"] = bgf.predict(4*3 ,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["aaa"] = bgf.predict(36*4 ,  # her bir müş için gerçekleştiğini nerden anlıyor
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'])

# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.

cltv_df.sort_values("exp_sales_3_month", ascending=False).head(10)

cltv_df.sort_values("exp_sales_6_month", ascending=False).head(10)

cltv_df.sort_values("aaa", ascending=False).head(10)
################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

plot_period_transactions(bgf)
plt.show(block=True)
# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

# müşterilerin ortlama bırakacakları değer
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary_cltv_avg'])
cltv_df.sort_values("exp_average_value", ascending=False).head(10)
# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6*4,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

# her bir gözlem birimi için customer_life time hesaplanacaktır
cltv.head()

cltv = cltv.reset_index()

#? birisi index de birisi sütunda buna ragmen brileştirme nasıl oluyor

cltv_final = cltv_df.merge(cltv, on="master_id", how="left")

cltv_final.head()

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.


cltv_final.sort_values(by="clv", ascending=False).head(20)

###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.
cltv_final["cltv_segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
cltv_final.groupby("cltv_segment").agg({"recency_cltv_weekly":"mean",
                                   "frequency":"mean",
                                   "monetary_cltv_avg" :"mean"})



