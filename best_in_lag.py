import glob
import pandas as pd

file_paths = glob.glob("result/rolling_corr_result_*.csv")
df_list = [pd.read_csv(f) for f in file_paths]
all_corr_df = pd.concat(df_list, ignore_index=True)

valid_corr = all_corr_df.dropna(subset=['correlation']).copy()

summary = valid_corr.groupby(['indicator_name', 'lag']).agg(
    mean_corr=('correlation', 'mean'),
    abs_mean_corr=('correlation', lambda x: x.abs().mean()),
    std_corr=('correlation', 'std'),           
    window_count=('window', 'count')       
).reset_index()

leading_indicators = summary[summary['lag'] > 0].copy()

best_per_indicator = leading_indicators.sort_values(
    by=['abs_mean_corr', 'std_corr'], 
    ascending=[False, True]
).groupby('indicator_name').first().reset_index()

print("=== Top 10 Best Indicator & Lag Combinations ===")
print(best_per_indicator[['indicator_name', 'lag', 'mean_corr', 'abs_mean_corr', 'std_corr']])