import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Make sure the project root (with src/) is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import itertools

from src.utils_lstm import compute_strategy_returns, compute_sharpe_ratio


# %% [markdown]
# ### Step 1: Load sample data

# %%
print(" ARIMAX Baseline Implementation for Stock Market Forecasting ")
print("=" * 70)

print("\n STEP 1: Loading Data ")
print("-" * 30)

try:
    df = pd.read_csv('data/core_proposal_daily_fixed_train.csv',
                     index_col=0,
                     parse_dates=True)
    

    drop_cols = [c for c in df.columns if c == 'CPI']
    if drop_cols:
        df = df.drop(columns=drop_cols)


    print(f"Dropped CPI columns from modeling dataset: {drop_cols}")
    print(f"✓ Loaded data: {df.shape}")
    print(f"✓ Date range: {df.index[0].strftime('%Y-%m-%d')} "
          f"to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Display basic info
    print(f"\nTarget variable (SP500_Returns) statistics:")
    returns = df['SP500_Returns'].dropna()
    print(f"  Mean: {returns.mean():.4f}%")
    print(f"  Std Dev: {returns.std():.4f}%")
    print(f"  Min: {returns.min():.4f}%")
    print(f"  Max: {returns.max():.4f}%")
    print(f"  Observations: {len(returns)}")
    
except FileNotFoundError:
    raise FileNotFoundError(
        "Sample data file not found. Expected: 'data/financial_dataset_daily.csv'"
    )

df_clean = df.dropna()
returns = df_clean['SP500_Returns']

print(f"\nData loaded successfully!")
print(f"Final dataset: {df_clean.shape} observations")

# %% [markdown]
# ### Step 2: Stationarity Analysis and Data Preparation

# %%

print("\n STEP 2: Stationarity Analysis ")
print("-" * 40)

def check_stationarity(timeseries, title):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    print(f"\n{title}:")
    print("-" * len(title))
    
    result = adfuller(timeseries.dropna())
    
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    
    if result[1] <= 0.05:
        print(" Series is stationary (reject null hypothesis)")
        return True
    else:
        print(" Series is non-stationary (fail to reject null hypothesis)")
        return False

# Check stationarity of returns (target variable)
is_stationary = check_stationarity(returns, "S&P 500 Returns Stationarity Test")

# Check stationarity of exogenous variables
# Exogenous candidates – script will only use those actually present
# Check stationarity of exogenous variables
# NOTE: we explicitly exclude CPI as a predictor
raw_exog_vars = ['VIX', 'Treasury_10Y', 'Yield_Spread', 'CPI']

# Drop CPI from the candidate list (even if present in the data)
exog_vars = [v for v in raw_exog_vars if v != 'CPI']

stationary_status = {}

for var in exog_vars:
    if var in df_clean.columns:
        stationary_status[var] = check_stationarity(df_clean[var], f"{var} Stationarity Test")
    else:
        print(f"Skipping {var}: not found in dataset.")


print(f"\n STATIONARITY SUMMARY:")
print(f"{'Variable':<15} {'Status':<12}")
print("-" * 30)
print(f"{'SP500_Returns':<15} {'Stationary' if is_stationary else 'Non-stationary'}")

for var, status in stationary_status.items():
    print(f"{var:<15} {'Stationary' if status else 'Non-stationary'}")


# %% [markdown]
# ### Step 3: Data Preprocessing

# %%
print("\n STEP 3: Data Preprocessing")
print("-" * 30)

df_model = df_clean.copy()

# Apply differencing to non-stationary exogenous variables
transformed_vars = []
for var in exog_vars:
    # Skip variables not actually present (e.g. Unemployment in core_proposal)
    if var not in df_model.columns:
        print(f"Skipping {var} - not in dataframe.")
        continue

    if not stationary_status.get(var, True):
        diff_name = f'{var}_diff'
        df_model[diff_name] = df_model[var].diff()
        transformed_vars.append(diff_name)
        print(f"✓ Applied differencing to {var}")
    else:
        transformed_vars.append(var)

# Remove NaNs created by differencing
df_model = df_model.dropna()

# EXTRA SAFETY: never use CPI-related variables even if they sneak in
transformed_vars = [
    v for v in transformed_vars
    if not (v == 'CPI' or v.startswith('CPI_'))
]

print(f"✓ Preprocessed data shape: {df_model.shape}")
print(f"✓ Final variables for ARIMAX (CPI excluded): {transformed_vars}")

# Prepare target and exogenous variables
y = df_model['SP500_Returns']
exog = df_model[transformed_vars]
print(f"Exogenous variables shape: {exog.shape}")




print(f"\n Model Data Summary:")
print(f"Target variable (y): {len(y)} observations")
print(f"Exogenous variables (X): {exog.shape}")
print(f"Date range: {df_model.index[0].strftime('%Y-%m-%d')} "
      f"to {df_model.index[-1].strftime('%Y-%m-%d')}")

# --------------------------------------------------------------------
# Train / test split (DO THIS BEFORE ORDER SELECTION to avoid leakage)
# --------------------------------------------------------------------
split_point = int(0.8 * len(y))
train_y = y[:split_point]
test_y = y[split_point:]

if exog is not None:
    train_exog = exog[:split_point]
    test_exog = exog[split_point:]
else:
    train_exog = None
    test_exog = None


print(f"\n Train/Test Split:")
print(f"Training set: {len(train_y)} observations "
      f"({train_y.index[0].strftime('%Y-%m-%d')} "
      f"to {train_y.index[-1].strftime('%Y-%m-%d')})")
print(f"Test set: {len(test_y)} observations "
      f"({test_y.index[0].strftime('%Y-%m-%d')} "
      f"to {test_y.index[-1].strftime('%Y-%m-%d')})")

# %% [markdown]
# ### Step 4: ARIMA Order Selection (p,d,q) — on TRAIN ONLY

# %%
print("\n STEP 4: ARIMA Order Selection")
print("-" * 40)

def find_optimal_arima_order(y_series, max_p=3, max_d=2, max_q=3):
    """
    Find optimal ARIMA order using AIC/BIC criteria on a given series.
    Here we pass TRAIN ONLY to avoid look-ahead bias.
    """
    print("Searching for optimal ARIMA(p,d,q) parameters (train only).")
    
    p_values = range(0, max_p + 1)
    d_values = range(0, max_d + 1)
    q_values = range(0, max_q + 1)
    pdq_combinations = list(itertools.product(p_values, d_values, q_values))
    
    results = []
    
    for pdq in pdq_combinations:
        try:
            model = ARIMA(y_series, order=pdq)
            fitted_model = model.fit()
            results.append({
                'order': pdq,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'llf': fitted_model.llf
            })
        except Exception:
            # Skip combinations that fail to converge
            continue
    
    if not results:
        print(" No valid ARIMA models found, falling back to (1,0,1)")
        return (1, 0, 1)
    
    results_df = pd.DataFrame(results)
    
    best_aic = results_df.loc[results_df['aic'].idxmin()]
    best_bic = results_df.loc[results_df['bic'].idxmin()]
    
    print(f"\n Top 5 models by AIC:")
    top_aic = results_df.nsmallest(5, 'aic')
    print(top_aic[['order', 'aic', 'bic']].to_string(index=False))
    
    print(f"\n Best model by AIC: ARIMA{best_aic['order']} "
          f"(AIC: {best_aic['aic']:.2f})")
    print(f" Best model by BIC: ARIMA{best_bic['order']} "
          f"(BIC: {best_bic['bic']:.2f})")
    
    return best_aic['order']

# IMPORTANT: use train_y here (no future information)
optimal_order = find_optimal_arima_order(train_y, max_p=3, max_d=2, max_q=3)
print(f"\n Selected ARIMA order: {optimal_order}")

# %% [markdown]
# ### Step 5: Fit ARIMAX Model

# %%
print(f"\n STEP 5: Fitting ARIMAX Model")
print("-" * 40)

print(f"Training set: {len(train_y)} observations "
      f"({train_y.index[0].strftime('%Y-%m-%d')} "
      f"to {train_y.index[-1].strftime('%Y-%m-%d')})")
print(f"Test set: {len(test_y)} observations "
      f"({test_y.index[0].strftime('%Y-%m-%d')} "
      f"to {test_y.index[-1].strftime('%Y-%m-%d')})")

print(f"\n Fitting ARIMAX{optimal_order} with exogenous variables.")
try:
    arimax_model = ARIMA(train_y, exog=train_exog, order=optimal_order)
    arimax_fitted = arimax_model.fit()


    
    print(" ARIMAX model fitted successfully!")
    print(f"\nModel Summary:")
    print(f"  AIC: {arimax_fitted.aic:.2f}")
    print(f"  BIC: {arimax_fitted.bic:.2f}")
    print(f"  Log-Likelihood: {arimax_fitted.llf:.2f}")
    print(f"  Observations: {arimax_fitted.nobs}")
    
    print(f"\n Parameter Estimates:")
    print(f"{'Parameter':<15} {'Coefficient':<12} "
          f"{'Std Error':<12} {'p-value':<10}")
    print("-" * 50)
    
    for param, coef, std_err, pval in zip(
        arimax_fitted.param_names,
        arimax_fitted.params,
        arimax_fitted.bse,
        arimax_fitted.pvalues
    ):
        significance = "***" if pval < 0.001 else \
                       "**" if pval < 0.01 else \
                       "*" if pval < 0.05 else ""
        print(f"{param:<15} {coef:<12.4f} {std_err:<12.4f} "
              f"{pval:<10.4f} {significance}")
    
    model_success = True

except Exception as e:
    print(f" Error fitting ARIMAX model: {str(e)}")
    print("Trying fallback ARIMA(1,0,1) model.")
    try:
        arimax_model = ARIMA(train_y, exog=train_exog, order=(1, 0, 1))
        arimax_fitted = arimax_model.fit()
        optimal_order = (1, 0, 1)
        model_success = True
        print(" Fallback model fitted successfully!")
    except Exception as e2:
        print(f" Fallback model also failed: {str(e2)}")
        model_success = False

print(f"is model_success?? -->  {model_success}")

# %% [markdown]
# ### Step 6: Model Evaluation and Forecasting

# %%
print(f"\n STEP 6: Model Evaluation and Forecasting")
print("-" * 50)

if model_success:
    print("Generating predictions...")
    
    # In-sample predictions
    in_sample_pred = arimax_fitted.fittedvalues
    
    # Out-of-sample predictions
    forecast_steps = len(test_y)
    if test_exog is not None:
        forecast = arimax_fitted.forecast(steps=forecast_steps, exog=test_exog)
    else:
        forecast = arimax_fitted.forecast(steps=forecast_steps)
    
    print(" Predictions generated successfully!")
    
    def calculate_metrics(actual, predicted, model_name):
        """Calculate evaluation metrics (including Sharpe ratio)."""
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = np.asarray(actual)[mask]
        predicted_clean = np.asarray(predicted)[mask]

        if len(actual_clean) == 0:
            return None

        mse = mean_squared_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_clean, predicted_clean)

        actual_direction = np.sign(actual_clean)
        predicted_direction = np.sign(predicted_clean)
        directional_accuracy = np.mean(actual_direction == predicted_direction)

        correlation = (
            np.corrcoef(actual_clean, predicted_clean)[0, 1]
            if len(actual_clean) > 1 else 0.0
        )

        # NEW: strategy returns & Sharpe ratio
        strategy_returns = compute_strategy_returns(
            predicted_clean, actual_clean
        )
        sharpe = compute_sharpe_ratio(strategy_returns)

        return {
            'Model': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Directional_Accuracy': directional_accuracy,
            'Correlation': correlation,
            'Sharpe_Ratio': sharpe,
            'Observations': len(actual_clean),
        }

    
    in_sample_metrics = calculate_metrics(
        train_y.values,
        in_sample_pred.values,
        f'ARIMAX{optimal_order} (In-Sample)'
    )
    
    out_sample_metrics = calculate_metrics(
        test_y.values,
        forecast.values,
        f'ARIMAX{optimal_order} (Out-of-Sample)'
    )
    
    print(f"\nPERFORMANCE EVALUATION:")
    print("=" * 70)

    if in_sample_metrics:
        print("IN-SAMPLE PERFORMANCE:")
        print(f"  RMSE: {in_sample_metrics['RMSE']:.4f}")
        print(f"  MAE: {in_sample_metrics['MAE']:.4f}")
        print(f"  Directional Accuracy: "
              f"{in_sample_metrics['Directional_Accuracy']:.1%}")
        print(f"  Correlation: {in_sample_metrics['Correlation']:.4f}")
        print(f"  Sharpe Ratio: {in_sample_metrics['Sharpe_Ratio']:.4f}")

    if out_sample_metrics:
        print("\nOUT-OF-SAMPLE PERFORMANCE:")
        print(f"  RMSE: {out_sample_metrics['RMSE']:.4f}")
        print(f"  MAE: {out_sample_metrics['MAE']:.4f}")
        print(f"  Directional Accuracy: "
              f"{out_sample_metrics['Directional_Accuracy']:.1%}")
        print(f"  Correlation: {out_sample_metrics['Correlation']:.4f}")
        print(f"  Sharpe Ratio: {out_sample_metrics['Sharpe_Ratio']:.4f}")

    
    print(f"\n MODEL DIAGNOSTICS:")
    print("-" * 30)
    
    residuals = arimax_fitted.resid.dropna()
    
    # Ljung-Box test
    try:
        lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
        lb_pval = lb_result['lb_pvalue'].iloc[-1]
        print(f"Ljung-Box test (p-value): {lb_pval:.4f}")
        if lb_pval > 0.05:
            print(" No significant residual autocorrelation")
        else:
            print(" Residual autocorrelation detected")
    except Exception as e:
        print(f" Could not perform Ljung-Box test: {str(e)}")
    
    # Normality test
    try:
        _, jb_pval = stats.jarque_bera(residuals)
        print(f"Jarque-Bera test (p-value): {jb_pval:.4f}")
        if jb_pval > 0.05:
            print(" Residuals appear normally distributed")
        else:
            print(" Residuals deviate from normality")
    except Exception as e:
        print(f" Could not perform normality test: {str(e)}")
    
    print("\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std Dev: {residuals.std():.4f}")
    print(f"  Min: {residuals.min():.4f}")
    print(f"  Max: {residuals.max():.4f}")
    
    print("\n SAMPLE PREDICTIONS (Last 10 test observations):")
    print("-" * 60)
    print(f"{'Date':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 60)
    
    for i in range(max(0, len(test_y) - 10), len(test_y)):
        date = test_y.index[i].strftime('%Y-%m-%d')
        actual = test_y.iloc[i]
        pred = forecast.iloc[i]
        error = actual - pred
        print(f"{date:<12} {actual:<10.3f} {pred:<10.3f} {error:<10.3f}")

else:
    in_sample_metrics = None
    out_sample_metrics = None
    print(" Model fitting failed. Cannot perform evaluation.")

print("\n ARIMAX BASELINE SUMMARY:")
print("=" * 50)
print(f"Model: ARIMAX{optimal_order}")
print(f"Training period: {train_y.index[0].strftime('%Y-%m-%d')} "
      f"to {train_y.index[-1].strftime('%Y-%m-%d')}")
print(f"Test period: {test_y.index[0].strftime('%Y-%m-%d')} "
      f"to {test_y.index[-1].strftime('%Y-%m-%d')}")
print(f"Exogenous variables: {list(exog.columns)}")

if model_success and out_sample_metrics:
    rmse = out_sample_metrics['RMSE']
    dir_acc = out_sample_metrics['Directional_Accuracy']
    corr = out_sample_metrics['Correlation']
    
    print("\nKey Performance Metrics:")
    print(f"  • Out-of-sample RMSE: {rmse:.4f}")
    print(f"  • Directional accuracy: {dir_acc:.1%}")
    print(f"  • Correlation: {corr:.4f}")
    
    print("\n BASELINE INTERPRETATION:")
    print("-" * 30)
    
    if rmse < 1.0:
        print(" Low prediction error - good baseline performance")
    elif rmse < 2.0:
        print(" Moderate prediction error - acceptable baseline")
    else:
        print(" High prediction error - challenging prediction task")
    
    if dir_acc > 0.55:
        print(" Above-random directional accuracy - model has predictive value")
    elif dir_acc > 0.45:
        print(" Near-random directional accuracy - limited predictive power")
    else:
        print(" Below-random directional accuracy - model may be misleading")
else:
    print(" Model evaluation incomplete - consider using ARIMA(1,0,1) fallback")

# %% [markdown]
# ### Step 7: Save Complete ARIMAX Implementation

# %%
print("\n STEP 7: Saving ARIMAX Implementation")
print("-" * 50)

# Ensure output directory exists
os.makedirs('reports/results', exist_ok=True)

results_summary = {
    'Model': f'ARIMAX{optimal_order}',
    'Training_Period': f"{train_y.index[0].strftime('%Y-%m-%d')} "
                       f"to {train_y.index[-1].strftime('%Y-%m-%d')}",
    'Test_Period': f"{test_y.index[0].strftime('%Y-%m-%d')} "
                   f"to {test_y.index[-1].strftime('%Y-%m-%d')}",
    'In_Sample_RMSE': in_sample_metrics['RMSE'] if in_sample_metrics else None,
    'Out_Sample_RMSE': out_sample_metrics['RMSE'] if out_sample_metrics else None,
    'Directional_Accuracy': (out_sample_metrics['Directional_Accuracy']
                             if out_sample_metrics else None),
    'Correlation': (out_sample_metrics['Correlation']
                    if out_sample_metrics else None),
    'Sharpe_Ratio': (out_sample_metrics['Sharpe_Ratio']
                     if out_sample_metrics else None),
    'AIC': arimax_fitted.aic if model_success else None,
    'BIC': arimax_fitted.bic if model_success else None,
}


results_df = pd.DataFrame([results_summary])
results_df.to_csv('reports/results/arimax_baseline_results.csv', index=False)
print("Saved ARIMAX results to 'reports/results/arimax_baseline_results.csv'")

if model_success and out_sample_metrics:
    # Strategy returns for ARIMAX: long/short on prediction sign
    strategy_returns = compute_strategy_returns(
        forecast.values, test_y.values
    )

    predictions_df = pd.DataFrame({
        'Date': test_y.index,
        'Actual': test_y.values,
        'Predicted': forecast.values,
        'Error': test_y.values - forecast.values,
        'Strategy_Returns': strategy_returns,
    })
    predictions_df.to_csv('reports/results/arimax_predictions.csv', index=False)
else:
    print("Skipping prediction CSV save because model evaluation was incomplete.")


print("\nARIMAX IMPLEMENTATION COMPLETE!")
print("="*50)
print("Files created (if model_success=True):")
print("1. 'arimax_baseline_results.csv' - Performance metrics")
print("2. 'arimax_predictions.csv' - Test set predictions")

if model_success and out_sample_metrics:
    print("\nKEY FINDINGS:")
    print(f"  • ARIMAX{optimal_order} model fitted successfully")
    print(f"  • Out-of-sample RMSE: {out_sample_metrics['RMSE']:.4f}")
    print(f"  • Directional accuracy: "
          f"{out_sample_metrics['Directional_Accuracy']:.1%}")
    print("  • Model diagnostics computed (autocorrelation, normality)")


# %% [markdown]
# ### Optional: Plot ARIMAX Results

# %%
def load_arimax_data():
    """Load ARIMAX results and predictions data."""
    try:
        results_df = pd.read_csv('reports/results/arimax_baseline_results.csv')

        model_name = results_df['Model'].iloc[0]
        rmse = results_df['Out_Sample_RMSE'].iloc[0]
        dir_acc = results_df['Directional_Accuracy'].iloc[0]
        correlation = results_df['Correlation'].iloc[0]
        sharpe = results_df['Sharpe_Ratio'].iloc[0] if 'Sharpe_Ratio' in results_df.columns else None


        predictions_df = pd.read_csv('reports/results/arimax_predictions.csv')
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])

        predictions_df['Actual_Direction'] = np.sign(predictions_df['Actual'])
        predictions_df['Predicted_Direction'] = np.sign(predictions_df['Predicted'])
        predictions_df['Direction_Correct'] = (
            predictions_df['Actual_Direction'] ==
            predictions_df['Predicted_Direction']
        )

        return {
            'model_name': model_name,
            'rmse': rmse,
            'directional_accuracy': dir_acc,
            'correlation': correlation,
            'sharpe': sharpe,
            'predictions': predictions_df,
        }


    except FileNotFoundError as e:
        print(f"Error loading ARIMAX data: {e}")
        print("Make sure the training section ran and saved the CSV files.")
        return None


def plot_arimax_results():
    data = load_arimax_data()
    if data is None:
        return

    model_name = data['model_name']
    rmse = data['rmse']
    dir_acc = data['directional_accuracy']
    correlation = data['correlation']
    sharpe = data.get('sharpe')
    predictions_df = data['predictions']


    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 10))

    # 1. Actual vs Predicted
    plt.subplot(2, 1, 1)
    plt.plot(predictions_df['Date'], predictions_df['Actual'],
             label='Actual Returns', linewidth=1.5)
    plt.plot(predictions_df['Date'], predictions_df['Predicted'],
             label='ARIMAX Prediction', linestyle='--', linewidth=1.5)
    plt.title(f'{model_name} - Actual vs Predicted Returns')
    plt.ylabel('Daily Returns (%)')
    plt.legend()

    # 2. Directional Accuracy over time
    plt.subplot(2, 1, 2)
    window = min(60, len(predictions_df))
    rolling_dir_acc = (
        predictions_df['Direction_Correct']
        .rolling(window=window)
        .mean()
    )
    plt.plot(predictions_df['Date'], rolling_dir_acc,
             label=f'Rolling Directional Accuracy ({window}-day)')
    plt.axhline(0.5, color='red', linestyle='--', label='Random (50%)')
    plt.ylim(0, 1)
    plt.ylabel('Directional Accuracy')
    plt.xlabel('Date')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nSUMMARY:")
    print(f"  Model: {model_name}")
    print(f"  Out-of-sample RMSE: {rmse:.4f}")
    print(f"  Directional accuracy: {dir_acc:.1%}")
    print(f"  Correlation: {correlation:.4f}")
    if sharpe is not None and not np.isnan(sharpe):
        print(f"  Sharpe ratio: {sharpe:.4f}")


if __name__ == "__main__":
    # If you want to auto-plot after running, uncomment:
    # plot_arimax_results()
    pass
