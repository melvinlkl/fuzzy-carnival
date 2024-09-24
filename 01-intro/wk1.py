import pandas as pd
import numpy as np

df = pd.read_csv('laptop.csv')

#Get number of records
print(len(df))

#Get distinct number of laptop brands
print(len(df.groupby(['Brand'])))

#Find missing columns
missing = df.isnull().sum()
print(missing)

#Get maximum price of Dell
dell_df = df[df['Brand'] == 'Dell']
print(dell_df[dell_df['Final Price'] == dell_df['Final Price'].max()])

#Find median of Screen
print(df['Screen'].median())

#Find mode of Screen
print(df['Screen'].mode())

#Fill null values with most frequent values
filled = df.fillna(df['Screen'].mode())
print(filled['Screen'].median())

#Get Innjoo laptop
innjoo_df = df[df['Brand'] == 'Innjoo']
in_df = innjoo_df[['RAM', 'Storage', 'Screen']]
x = in_df.to_numpy()
x_t = x.transpose()
x_t_x = np.dot(x_t, x)
inv_xtx = np.linalg.inv(x_t_x)
y = np.array([1100, 1300, 800, 900, 1000, 1100])
w1 = np.dot(inv_xtx, x_t)
w = np.dot(w1,y)
print(w.sum())
