from clickhouse_driver import Client
import logging
import pandas as pd
from src.config import host

def insert_to_db():
    client = Client(host= host, port= 9000, user= 'user1', password= '123456')
    # client.execute('show databases')

    logging.info("Opened database successfully")
    client.execute("""
    CREATE TABLE IF NOT EXISTS testing.winprice_estimation
    (
        date DateTime,
        price Float64,
        bedrooms Float32,
        bathrooms Float32,
        sqft_living Int64,
        sqft_lot Int64,
        floors Float32,
        waterfront Int64,
        view Int64,
        condition Int64,
        sqft_above Int64,
        sqft_basement Int64,
        yr_built Int64,
        yr_renovated Int64,
        street String,
        city String,
        statezip String,
        country String,
        weekday String
    )
    ENGINE = MergeTree()
    ORDER BY date;
    """)

    logging.info("Table created successfully")


    df = pd.read_csv("./data/data.csv",
                    usecols= lambda x: x != 'Id')
    df['date'] = pd.to_datetime(df['date'])

    df['weekday'] = pd.to_datetime(df.date).dt.dayofweek.apply(lambda x: x+1).astype(str)

    client.execute(f"INSERT INTO testing.winprice_estimation VALUES", df.to_dict('records'))
    logging.info("Records created successfully")
    client.disconnect()


if __name__ == "__main__":
    insert_to_db()

