import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")
print("Dataset info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

df = df[(df['Model Year'] >= 1990) & (df['Model Year'] <= 2025)]
print("\nShape after cleaning:", df.shape)

reduced_df = df[[
    'Make',
    'Model',
    'Model Year',
    'Electric Range',
    'Base MSRP',
    'County',
    'City',
    'Electric Vehicle Type'
]]

print("\nShape after reduction:", reduced_df.shape)
print(reduced_df.head())

scaler = MinMaxScaler()
cols_to_scale = ['Electric Range', 'Base MSRP']

reduced_df.loc[:, cols_to_scale] = scaler.fit_transform(reduced_df[cols_to_scale])
print("\nAfter normalization:")
print(reduced_df[cols_to_scale].describe())

range_bins = [0.0, 0.1, 0.5, 1.0]
range_labels = ['Low', 'Medium', 'High']

reduced_df.loc[:, 'Range Category'] = pd.cut(
    reduced_df['Electric Range'],
    bins=range_bins,
    labels=range_labels,
    include_lowest=True
)

print("\nElectric Range categories:")
print(reduced_df['Range Category'].value_counts())
