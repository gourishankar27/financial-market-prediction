import pandas as pd
import ast

df = pd.read_csv('experiments/wf_monthly_weekly_enc12/rolling_results_full.csv')

# Parse nested dicts
df['dir_acc'] = df['financial_metrics'].apply(lambda x: ast.literal_eval(x)['directional_accuracy'])
df['sharpe'] = df['financial_metrics'].apply(lambda x: ast.literal_eval(x)['sharpe_ratio'])
df['healthy'] = df['mode_stats'].apply(lambda x: ast.literal_eval(x)['healthy_pct'])


print(f"Folds: {len(df)}")
print(f"Dir Acc: {df['dir_acc'].mean():.3f} ± {df['dir_acc'].std():.3f}")
print(f"Sharpe:  {df['sharpe'].mean():.3f} ± {df['sharpe'].std():.3f}")
print(f"Healthy: {df['healthy'].mean():.1f}% ± {df['healthy'].std():.1f}%")
print(f"\nBy year:")
print(df.groupby('test_year')[['dir_acc', 'sharpe', 'healthy']].mean().round(3))

# Aggregate monthly to annual for fair comparison
annual = df.groupby('test_year').agg({
    'dir_acc': 'mean',
    'sharpe': 'mean'  # Note: should really compound returns, not average Sharpe
}).round(3)
print(annual)