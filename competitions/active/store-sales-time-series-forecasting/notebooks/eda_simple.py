import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_path = 'data/store-sales-time-series-forecasting/train.csv'
stores_path = 'data/store-sales-time-series-forecasting/stores.csv'

train_df = pd.read_csv(train_path)
stores_df = pd.read_csv(stores_path)

# Print basic info
print("Train DataFrame Info:")
print("Shape:", train_df.shape)
print("Data types:\n", train_df.dtypes)
print
