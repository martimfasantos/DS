from pandas import read_csv, to_datetime
from pandas.plotting import register_matplotlib_converters

# Encoding of date variable from dd/mm/yyyy to ddmmyyyy
register_matplotlib_converters()
file_tag = 'drought'
file_path = '../datasets/classification/drought.csv'
data = read_csv(file_path, na_values="na", sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
data['date'] = to_datetime(data['date'], dayfirst=True).dt.strftime('%02d%02m%Y') 
data.to_csv(f'data/variables_encoding/{file_tag}_variables_encoding.csv')
