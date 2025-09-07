import pandas as pd
from pyspark.sql import SparkSession

def load_data():
    url = "https://raw.githubusercontent.com/liaoyuhua/open-time-series-datasets/master/exchange-rate/exchange_rate.txt"
    df = pd.read_csv(url, sep=",", header=None, names=["Timestamp", "Value"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Value"] = (df["Value"] - df["Value"].min()) / (df["Value"].max() - df["Value"].min())
    return df

def distribute_data(df):
    spark = SparkSession.builder.appName("LSTM_Distributed").getOrCreate()
    spark_df = spark.createDataFrame(df)
    return spark_df.repartition(4)

if __name__ == "__main__":
    df = load_data()
    spark_df = distribute_data(df)
    spark_df.show(5)
