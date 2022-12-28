from pandas import read_csv, to_datetime
from pandas.plotting import register_matplotlib_converters
import math
import numpy as np

# Encoding of date variable from dd/mm/yyyy to ddmmyyyy
register_matplotlib_converters()
file_tag = 'drought'
file_path = '../datasets/classification/drought.csv'
data = read_csv(file_path, na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
#data['date'] = to_datetime(data['date'], dayfirst=True).dt.strftime('%04Y%02m%02d')

data['month'] = 0
data['year'] = 0

# Month
for ind in data.index:
    date = data['date'][ind].split('/')
    day = date[0]
    month = date[1]
    year = date[2]
    data['date'][ind] = int(day + month + year)
    data['month'][ind] = int(month)
    data['year'][ind] = int(year)

data["month_norm"] = 2 * math.pi * data["month"] / data["month"].max()
data["cos_month"] = np.cos(data["month_norm"])
data["sin_month"] = np.sin(data["month_norm"])

data.drop(columns='month_norm', inplace=True)
data.drop(columns='month', inplace=True)

data.to_csv(f'data/variables_encoding/{file_tag}_variables_encoding.csv', index=False)
