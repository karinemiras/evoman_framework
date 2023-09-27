import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




# Define the name of the CSV file you want to access
csv_file_name = ['/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-0.csv', '/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-1.csv', '/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-2.csv', 
                 '/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-3.csv', '/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-4.csv', '/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-5.csv',
                 '/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-6.csv', '/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-7.csv'
                 , '/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-8.csv', '/Users/joanacostaesilva/Desktop/CLS/Evolutionary Computing/evoman_framework-group-33/7_Fixed_Structure/stats-fitness-7-9.csv']



dfs = [pd.read_csv(file) for file in csv_file_name]


# Combine the DataFrames vertically
combined_df = pd.concat(dfs, keys=['Stats-Fitness 1', 'Stats-Fitness 2', 'Stats-Fitness 3',
                                   'Stats-Fitness 4','Stats-Fitness 5','Stats-Fitness 6','Stats-Fitness 7','Stats-Fitness 8','Stats-Fitness 9','Stats-Fitness 10'])

# Group by 'Generation' and calculate statistics
stats_df = combined_df.groupby(['generation']).agg({'max_fitness': ['mean', 'std'], 'mean_fitness': ['mean', 'std']})


# Rename the columns for clarity
stats_df.columns = ['Max Fitness Mean', 'Max Fitness Std', 'Mean Fitness Mean', 'Mean Fitness Std']


# Reset the index to make 'Experiment' and 'Generation' regular columns
stats_df.reset_index(inplace=True)

# Save the resulting DataFrame to a new CSV file if needed:
stats_df.to_csv('Experiment_enemy7_Fixed_structure.csv', index=False)

print(stats_df)

# Line-plot with the average/std (for the mean and the maximum) of the fitness from stats_df

sns.lineplot(x='generation', y='Max Fitness Mean', data=stats_df, label='Max Fitness Mean')
#sns.lineplot(x='generation', y='Max Fitness Std', data=stats_df, label='Max Fitness Std')
sns.lineplot(x='generation', y='Mean Fitness Mean', data=stats_df, label='Mean Fitness Mean')
#sns.lineplot(x='generation', y='Mean Fitness Std', data=stats_df, label='Mean Fitness Std')




max_fit_mean = stats_df['Max Fitness Mean'].values
max_fit_std = stats_df['Max Fitness Std'].values

mean_fit_mean = stats_df['Mean Fitness Mean'].values
mean_fit_std = stats_df['Mean Fitness Std'].values

x = stats_df['generation'].values
plt.fill_between(x, max_fit_mean - max_fit_std, max_fit_mean + max_fit_std, alpha=0.5, color='gray', label='Mean ± Std Dev')
plt.fill_between(x, mean_fit_mean - mean_fit_std, mean_fit_mean + mean_fit_std, alpha=0.5, color='gray', label='Mean ± Std Dev')


plt.title('Fitness progression')
plt.xlabel('Generations')
plt.ylabel('Fitness')
# Show the plot
plt.show()


