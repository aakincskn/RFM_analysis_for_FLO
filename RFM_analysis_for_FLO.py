###############################################################
# Customer Segmentation with RFM
###############################################################


# Business Problem
###############################################################
# FLO wants to segment its customers and determine marketing strategies according to these segments.
# For this purpose, customers' behaviors will be defined and groups will be formed according to these behavioral clusters.

###############################################################
# Dataset Story
###############################################################

#The data set consists of information obtained from the past shopping behavior of customers
# who made their last purchases as OmniChannel (both online and offline shoppers) in 2020 - 2021.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel :  The channel where the last purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_date : Date of the customer's last purchase
# last_order_date_online :  The date of the last purchase made by the customer on the online platform
# last_order_date_offline : Date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : Total number of online purchases made by the customer
# order_num_total_ever_offline : Total number of offline purchases made by the customer
# customer_value_total_ever_offline : Total price paid by the customer for offline purchases
# customer_value_total_ever_online : Total price paid by the customer for online shopping
# interested_in_categories_12 : List of categories in which the customer shopped in the last 12 months

###############################################################
# DUTIES
###############################################################

# DUTY 1: Data Understanding and Preparation
           # 1. Read flo_data_20K.csv data.
           # 2. In the data set
                     # a. Top 10 observations,
                     # b.  Variable names,
                     # c. Descriptive statistics,
                     # d. Empty value,
                     # e. Examine variable types.
           # 3. Omnichannel means that customers shop from both online and offline platforms. Create new variables
           # for each customer's total number of purchases and spend.
           # 4. Examine variable types. Change the type of variables that express date to date.
           # 5. See the distribution of the number of customers, average number of products purchased and average spending across shopping channels.
           # 6. List the top 10 most profitable customers.
           # 7. List the top 10 customers who placed the most orders.
           # 8. Functionalize the data preparation process.

# DUTY 2: Calculation of RFM Metrics

# DUTY 3: Calculation of RF and RFM Scores

# DUTY 4: Segmental Identification of RF Scores

# DUTY 5: Action Time!
           # 1. Examine the recency, frequnecy and monetary averages of the segments.
           # 2. With the help of RFM analysis, find the customers in the relevant profile for 2 cases and save the customer ids to csv.
                   # a. FLO includes a new women's shoe brand in its organization.
                    # The product prices of the brand it includes are above the general customer preferences.
                    # For this reason, it is desired to contact customers who will be interested in promoting the brand and product sales. Loyal customers (champions, loyal_customers),
                    # people who shop on average over 250 TL and in the women's category are the customers to be specially contacted.
                    # Save the id numbers of these customers in csv file as new_brand_target_customer_id.cvs.
                   # b. Up to 40% discount is planned for Men's and Children's products.
                   # With this discount, customers who are interested in the relevant categories, who have been good customers in the past but have not been shopping for a long time,
                   # customers who should not be lost, dormant customers and new customers are to be specifically targeted.
                   # Save the ids of the customers in the appropriate profile to csv file as discount_target_customer_ids.csv.


# DUTY 6: Functionalize the whole process.

###############################################################
# DUTY 1: Data Understanding and Preparation
###############################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

cltv



# 1. Read flo_data_20K.csv data.
df_ = pd.read_csv("/Users/macbook/Desktop/Miuul/2.hafta crm/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy()

# 2. In the data set
                     # a. Top 10 observations,
                     # b.  Variable names,
                     # c. Descriptive statistics,
                     # d. Empty value,
                     # e. Examine variable types.
#a
df.head(10)
#b
df.columns
#c
df.describe().T
#d
df.isnull().sum()
#e
df.dtypes


# 3. Omnichannel means that customers shop from both online and offline platforms. Create new variables
           # for each customer's total number of purchases and spend.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# 4. Examine variable types. Change the type of variables that express date to date.
date_columns = df.columns[df.columns.str.contains("date")]

df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# 5. See the distribution of the number of customers, average number of products purchased and average spending across shopping channels.
df.groupby("order_channel").agg({"master_id": lambda x: x.nunique(),
                                 "order_num_total" : lambda x: x.sum(),
                                 "customer_value_total" : lambda x: x.sum()})

# 6. List the top 10 most profitable customers.
df.sort_values("customer_value_total", ascending= False).head(10)

# 7. List the top 10 customers who placed the most orders.
df.sort_values("order_num_total", ascending= False).head(10)


# 8. Functionalize the data preparation process.
def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df

#########################################
# DUTY 2: Calculation of RFM Metrics
#################
# Analysis date(today_date, variable)  2 days after the date of the last purchase in the dataset
analysis_date = dt.datetime(2021,6,1)

df["last_order_date"].max() # 2021-05-30

# a new rfm dataframe with customer_id, recency, frequnecy and monetary values
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

###############################################################
# DUTY 3: Calculating RF and RFM Scores
###############################################################

#  Convert Recency, Frequency and Monetary metrics into scores between 1-5 with the help of qcut and
# save these scores as recency_score, frequency_score and monetary_score
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels= [5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method= "first"), 5, labels= [1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels= [1, 2, 3, 4, 5])

# expressing recency_score and frequency_score as a single variable and saving as RF_SCORE
rfm['customer_id'].nunique()

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

# Express recency_score and frequency_score and monetary_score as a single variable and save as RFM_SCORE
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

###############################################################
# DUTY 4: Segmental Identification of RF Scores
###############################################################

# Defining segments to make the generated RFM scores more explainable and translating RF_SCORE into segments with the help of the defined seg_map

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
    r'5[4-5]': 'champions'
}


rfm["segment"] = rfm['RF_SCORE'].replace(seg_map, regex=True)

###############################################################
# DUTY 5: Action time!
###############################################################

# 1. Examine the recency, frequnecy and monetary averages of the segments.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#                          recency       frequency       monetary
#                        mean count      mean count     mean count
# segment
# about_to_sleep       113.79  1629      2.40  1629   359.01  1629
# at_Risk              241.61  3131      4.47  3131   646.61  3131
# cant_loose           235.44  1200     10.70  1200  1474.47  1200
# champions             17.11  1932      8.93  1932  1406.63  1932
# hibernating          247.95  3604      2.39  3604   366.27  3604
# loyal_customers       82.59  3361      8.37  3361  1216.82  3361
# need_attention       113.83   823      3.73   823   562.14   823
# new_customers         17.92   680      2.00   680   339.96   680
# potential_loyalists   37.16  2938      3.30  2938   533.18  2938
# promising             58.92   647      2.00   647   335.67   647







# 2. With the help of RFM analysis, find the customers in the relevant profile for 2 cases and save the customer ids to csv.

#a. FLO includes a new women's shoe brand in its organization.
#The product prices of the brand it includes are above the general customer preferences.
#For this reason, it is desired to contact the customers who will be interested in the promotion of the brand and product sales.
#These customers are planned to be loyal and female category shoppers. Save the id numbers of the customers in csv file as new_brand_target_customer_id.csv.


target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

rfm.head()


# b. Up to 40% discount is planned for Men's and Children's products.
# With this discount, we want to specifically target customers who are interested in the relevant categories,
# who have been good customers in the past but have not shopped for a long time and new customers.
# Save the ids of the customers in the appropriate profile to csv file as discount_target_customer_ids.csv.


target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)


###############################################################
# BONUS
###############################################################

def create_rfm(dataframe):
    # Data preparation
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)


    # CALCULATION OF RFM METRICS
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    rfm = pd.DataFrame()
    rfm["customer_id"] = dataframe["master_id"]
    rfm["recency"] = (analysis_date - dataframe["last_order_date"]).astype('timedelta64[D]')
    rfm["frequency"] = dataframe["order_num_total"]
    rfm["monetary"] = dataframe["customer_value_total"]

    # CALCULATION OF RF and RFM SCORES
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))


    # NOMENCLATURE OF SEGMENTS
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
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    return rfm[["customer_id", "recency","frequency","monetary","RF_SCORE","RFM_SCORE","segment"]]

rfm_df = create_rfm(df)















































































