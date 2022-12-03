from pandas import DataFrame, concat, read_csv, to_datetime, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, bar_chart, get_variable_types
from sklearn.preprocessing import OneHotEncoder
from numpy import number
from matplotlib.pyplot import figure, savefig, show
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

#encoding of date varaible from dd/mm/yyyy to ddmmyyyy
register_matplotlib_converters()
data = read_csv('../datasets/classification/drought.csv', na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
data['date'] = to_datetime(data['date'], dayfirst=True).dt.strftime('%02d%02m%Y') 
data.to_csv(f'../datasets/data/climate_data_variables_encoding.csv')


