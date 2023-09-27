import csv
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import os 

experiment_names = ['7_Fixed_Structure','7_Fixed_Structure','7_Fixed_Structure','7_Fixed_Structure','7_Fixed_Structure','7_Fixed_Structure']

original_enemy =[7,7,7,7,7,7]
#run_number = 9
new_enemy = [7,7,7,7,7,7]



df_csv_append = pd.DataFrame()
for experiment_name in experiment_names:
    for i in range(10):
        with open(os.path.join(experiment_name, f'individual_gain-{original_enemy[experiment_names.index(experiment_name)]}-{i}-{new_enemy[experiment_names.index(experiment_name)]}.csv'), 'rb') as input_file:
            df = pd.read_csv(input_file)
            df_csv_append = df_csv_append.append(df, ignore_index=True)
            
print(df_csv_append)


_fig = plt.figure(figsize =(8, 5)) 
# Making a plot 
_data_1 = df_csv_append['Mean'][:10]
_data_2 = df_csv_append['Mean'][10:20]
_data_3 = df_csv_append['Mean'][20:30]
_data_4 = df_csv_append['Mean'][30:40]
_data_5 = df_csv_append['Mean'][40:50]
_data_6 = df_csv_append['Mean'][50:60]
_data = [_data_1, _data_2, _data_3, _data_4, _data_5, _data_6] 

plt.title("Individual gain per enemy & per algorithm") 
plt.boxplot(_data,patch_artist=True,labels=['Fixed1','Flexible1','Fixed7','Flexible7','Fixed8','Flexible8'])

#plt.savefig('Example_Boxplot.png')

# display the plot 
plt.show() 