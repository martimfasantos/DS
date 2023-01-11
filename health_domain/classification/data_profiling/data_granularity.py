from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types, bar_chart, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show, tight_layout

register_matplotlib_converters()
filename = '../datasets/diabetic_data.csv'
data = read_csv(filename, na_values='?')


# -------------------------------- #
# Histograms for numeric variables #
# -------------------------------- #

variables = get_variable_types(data)['Numeric']
if [] == variables:
    raise ValueError('There are no numeric variables.')

rows = len(variables)
bins = (5, 10, 50)
cols = len(bins)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(rows):
    for j in range(cols):
        axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(data[variables[i]].values, bins=bins[j])
tight_layout()
savefig('./images/granularity_study_numeric.png')
# show()

# --------------------------------- #
# Histograms for symbolic variables #
# --------------------------------- #

def has_letter(word: str):
    for char in word:
        if char.isalpha():
            return True
    return False

def do_group_1(counts: list) -> list:
    new_counts = {'injury': 0, 'circulatory': 0, 'neoplasms': 0, 
                  'other': 0, 'diabetes': 0, 'genitourinary': 0,
                  'musculoskeletal': 0, 'digestive': 0, 'respiratory': 0 }
    for item in counts:
        if (has_letter(str(item[0]))):
            new_counts['other'] += item[1]
        else:
            code = int(str(item[0][0:3]))
            if   (code >= 800) and (code <= 999):
                new_counts['injury'] += item[1]
            elif ((code >= 390) and (code <= 459)) or (code == 785):
                new_counts['circulatory'] += item[1]
            elif (code >= 140) and (code <= 239):
                new_counts['neoplasms'] += item[1]
            elif (code == 250):
                new_counts['diabetes'] += item[1]
            elif ((code >= 580) and (code <= 629)) or (code == 788):
                new_counts['genitourinary'] += item[1]
            elif (code >= 710) and (code <= 739):
                new_counts['musculoskeletal'] += item[1]
            elif ((code >= 520) and (code <= 579)) or (code == 787):
                new_counts['digestive'] += item[1]
            elif ((code >= 460) and (code <= 519)) or (code == 786):
                new_counts['respiratory'] += item[1]
            else:
                new_counts['other'] += item[1]
    return new_counts

def do_group_2(counts: list) -> list:
    new_counts = {'Caucasian': 0, 'AfricanAmerican': 0, 'Other': 0 }
    for item in counts:
        try:
            new_counts[item[0]] += item[1]
        except:
            new_counts['Other'] += item[1]
    return new_counts

def do_group_3(counts: list) -> list:
    new_counts = {'0,30]': 0, '(30,60]': 0, '(60, 100)': 0}
    for item in counts:
        if (item[0] in ('[0-10)', '[10-20)', '[20-30)')):
            new_counts['0,30]'] += item[1]
        elif (item[0] in ('[30-40)', '[40-50)', '[50-60)')):
            new_counts['(30,60]'] += item[1]
        else:
            new_counts['(60, 100)'] += item[1]
    return new_counts

def do_group_4(counts: list) -> list:
    new_counts = {'[0,50]': 0, '(50,125]': 0, '>100': 0 }
    for item in counts: 
        if (item[0] in ('[0-25)', '[25-50)')):
            new_counts['[0,50]'] += item[1]
        elif (item[0] in ('[50-75)', '[75-100)', '[100-125)')):
            new_counts['(50,125]'] += item[1]
        else:
            new_counts['>100'] += item[1]
    return new_counts

def do_group_5(counts: list) -> list:
    new_counts = {'Family/GeneralPractice': 0, 'InternalMedicine': 0,
                  'Other': 0, 'Surgery': 0, 'Cardiology': 0 }
    for item in counts:
        try:
            new_counts[item[0]] += item[1]
        except:
            if (item[0][0:7] == 'Surgery'):
                new_counts['Surgery'] += item[1]
            else:
                new_counts['Other'] += item[1]
    return new_counts

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# variables = get_variable_types(data)['Symbolic']
# if [] == variables:
#     raise ValueError('There are no numeric variables.')

# rows = len(variables)
# bins = (5, 10, 50)
# cols = len(bins)
# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
# for i in range(rows):
#     for j in range(cols):
#         axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
#         axs[i, j].set_xlabel(variables[i])
#         axs[i, j].set_ylabel('Nr records')
#         axs[i, j].hist(data[variables[i]].values, bins=bins[j])
# savefig('./images/granularity_study_symbolic.png')
# # show()

# symbolic_vars = get_variable_types(data)['Symbolic']
# if [] == symbolic_vars:
#     raise ValueError("There are no symbolic variables.")

# rows, cols = choose_grid(len(symbolic_vars))
# fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT*4, rows*HEIGHT*3), squeeze=False)
# i, j = 0, 0
# for n in range(len(symbolic_vars)):
#     counts = data[symbolic_vars[n]].value_counts()
#     if (counts.name in ('age', 'weight')):
#         #counts.sort_index()
#         counts = counts.reset_index().values.tolist()
#         counts.sort(key=lambda x: int(x[0].split('-')[0][1:]))
#         x, y = [], []
#         for el in counts:
#             x.append(el[0])
#             y.append(el[1])
#         bar_chart(x, y, ax=axs[i, j], title='Histogram for %s' %symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False, rotation=45)
#     else:
#         bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s' %symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False, rotation=45)
#     i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
# savefig('./images/granularity_study_symbolic.png')
# show()

group_1 = ('diag_1', 'diag_2', 'diag_3') # diagnoses group
group_2 = ('race')
group_3 = ('age')
group_4 = ('weight')
group_5 = ('medical_specialty')
    
symbolic_vars = get_variable_types(data)['Symbolic']
if [] == symbolic_vars:
    raise ValueError("There are no symbolic variables.")
    
rows, cols = choose_grid(len(symbolic_vars) + 7)
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT*4, rows*HEIGHT*3), squeeze=False)
i, j = 0, 0
n_graphs = 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts() # counts for each variable value
    if (counts.name in ('age', 'weight')):
        neww_counts = counts.reset_index().values.tolist()
        neww_counts.sort(key=lambda x: int(x[0].split('-')[0][1:]))
        x, y = [], []
        for el in neww_counts:
            x.append(el[0])
            y.append(el[1])
        bar_chart(x, y, ax=axs[i, j], title='Histogram for %s' %symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False, rotation=45)
    else:
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s' %symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False, rotation=45)
    i, j = (i + 1, 0) if (n_graphs + 1) % cols == 0 else (i, j + 1)
    n_graphs += 1  
    
    # Created Groups 
    bins = 3
    counts = counts.reset_index().values.tolist()

    if (symbolic_vars[n] in group_1):
        new_counts = do_group_1(counts)
        bins = 9
    elif (symbolic_vars[n] in group_2):
        new_counts = do_group_2(counts)
    elif (symbolic_vars[n] in group_3):
        new_counts = do_group_3(counts)
    elif (symbolic_vars[n] in group_4):
        new_counts = do_group_4(counts)
    elif (symbolic_vars[n] in group_5):
        new_counts = do_group_5(counts)
        bins = 5
    else:
        continue
    
    x, y = [], []
    for key in new_counts:
        x.append(key)
        y.append(new_counts[key])
    
    bar_chart(x, y, ax=axs[i, j], title='Histogram for %s WITH %d BINS' %(symbolic_vars[n], bins), xlabel=symbolic_vars[n], ylabel='nr records', percentage=False, rotation=45)
    i, j = (i + 1, 0) if (n_graphs + 1) % cols == 0 else (i, j + 1)
    n_graphs += 1
tight_layout()    
savefig('./images/granularity_study_symbolic.png', dpi=90)
      
# ----------------------------- #
# Histograms for date variables #
# ----------------------------- #

variables = get_variable_types(data)['Date']
if [] == variables:
    fig, axs = subplots(figsize=(8, 4)) 
    axs.set_title("No Date Variables")          # Do any Matplotlib customization you like
else:
    rows = len(variables)
    bins = (5, 10, 50)
    cols = len(bins)
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
            axs[i, j].set_xlabel(variables[i])
            axs[i, j].set_ylabel('Nr records')
            axs[i, j].hist(data[variables[i]].values, bins=bins[j])
tight_layout()
savefig('./images/granularity_study_date.png')
# show()