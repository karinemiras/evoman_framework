from scipy import stats
import pandas as pd

data_NEAT = pd.read_csv('final_plot_data/boxplot_NEAT_gains.csv')
data_NORMAL = pd.read_csv('final_plot_data/boxplot_data_normal_gains.csv')
data_MEMORY = pd.read_csv('final_plot_data/boxplot_data_memory_gains.csv')

enemies = [1,5, 8]

results = pd.DataFrame([], columns = ['NORMAL > NEAT', 'MEMORY > NEAT', 'MEMORY > NORMAL'])


for idx, enemy in enumerate(enemies):
    rv1 = data_NEAT[f'enemy_{enemy}']
    rv2 = data_NORMAL[f'enemy_{enemy}']
    rv3 = data_MEMORY[f'enemy_{enemy}']

    result1 = stats.ttest_ind(rv2, rv1 , equal_var=False, alternative = 'greater')[1]
    result2 = stats.ttest_ind(rv3, rv1, equal_var=False, alternative='greater')[1]
    result3 = stats.ttest_ind(rv3, rv2, equal_var=False, alternative='greater')[1]

    results.loc[idx, "NORMAL > NEAT"] = result1
    results.loc[idx, "MEMORY > NEAT"] = result2
    results.loc[idx, "MEMORY > NORMAL"] = result3

print(results)