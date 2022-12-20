from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, title, savefig, show, tight_layout
from seaborn import heatmap
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types, bar_chart_fs

file_tag = 'diabetic_data'
file_name = f'data/scaling/{file_tag}_scaled_zscore.csv'
data = read_csv(file_name)

# discard all id's
data.pop('encounter_id')
data.pop('admission_type_id')
data.pop('discharge_disposition_id')
data.pop('admission_source_id')
data.pop('patient_nbr')

'''We chose not to drop these features since we obtained better performance with them'''
# data.pop('number_outpatient')
# data.pop('number_inpatient')
# data.pop('number_emergency')


# ---------------------------- #
# Dropping Redundant Variables #
# ---------------------------- #

THRESHOLD = 0.9

def select_redundant(corr_mtx, threshold: float) -> tuple[dict, DataFrame]:
    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_2drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_2drop[el] = el_corr.index
    return vars_2drop, corr_mtx

drop, corr_mtx = select_redundant(data.corr(), THRESHOLD)
# print("Redundancies: ", drop.keys())

if corr_mtx.empty:
    raise ValueError('Matrix is empty.')

figure(figsize=[12, 12])
heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
title('Filtered Correlation Analysis')
tight_layout()
savefig(f'images/feature_selection/{file_tag}_filtered_correlation_analysis_{THRESHOLD}.png')


def drop_redundant(data: DataFrame, vars_2drop: dict) -> DataFrame:
    sel_2drop = []
    # print(vars_2drop.keys())
    for key in vars_2drop.keys():
        if key not in sel_2drop:
            for r in vars_2drop[key]:
                if r != key and r not in sel_2drop:
                    sel_2drop.append(r)
    #print('Variables to drop: ', sel_2drop)
    df = data.copy()
    for var in sel_2drop:
        df.drop(labels=var, axis=1, inplace=True)
    return df

df = drop_redundant(data, drop)


def select_low_variance(data: DataFrame, threshold: float) -> list:
    lst_variables = []
    lst_variances = []
    for el in data.columns:
        value = data[el].var()
        if value <= threshold:
            lst_variables.append(el)
            lst_variances.append(value)

    # print(len(lst_variables), lst_variables)
    figure(figsize=[16, 10])
    bar_chart_fs(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance', rotation=True)
    tight_layout()
    savefig(f'images/feature_selection/{file_tag}_filtered_variance_analysis.png')
    return lst_variables

numeric = get_variable_types(data)['Numeric']
vars_2drop = select_low_variance(data[numeric], 0.1)
# print(vars_2drop)

'''We chose not to drop these features since we obtained better performance with them'''
# for var in vars_2drop:
#     df.drop(labels=var, axis=1, inplace=True)

df.to_csv(f'data/feature_selection/{file_tag}_selected.csv', index=False)
