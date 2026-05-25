import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3



def compute_strategy_returns(predicted_returns, actual_returns):
    """
    Build a simple long/short strategy based on the sign of the predictions.
    Assumes both inputs are 1D arrays of (daily) returns.
    Strategy:
        - Long if predicted return > 0
        - Short if predicted return < 0
        - Flat if predicted return == 0
    """
    predicted_returns = np.asarray(predicted_returns).reshape(-1)
    actual_returns = np.asarray(actual_returns).reshape(-1)

    if predicted_returns.shape != actual_returns.shape:
        raise ValueError("predicted_returns and actual_returns must have the same shape")

    signal = np.sign(predicted_returns)  # -1, 0, +1
    strategy_returns = signal * actual_returns
    return strategy_returns



def compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Annualized Sharpe ratio.
    `returns` should be per-period (e.g. daily) strategy returns.
    """
    returns = np.asarray(returns).reshape(-1)

    if returns.size < 2:
        return np.nan

    # convert annual risk-free to per-period (optional; often left as 0)
    excess_returns = returns - risk_free_rate / periods_per_year

    std = np.std(excess_returns, ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan

    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / std
    return sharpe


def load_arimax_baseline(results_path: str = 'reports/results/arimax_baseline_results.csv'):
    """
    Load ARIMAX baseline metrics saved by train_arimax.py.

    Returns
    -------
    dict or None
        {
            'model_name': str,
            'rmse': float,
            'dir_acc': float,      # directional accuracy in [0, 1]
            'corr': float
        }
        or None if the file is missing / unreadable.
    """
    if not os.path.exists(results_path):
        print(f"[ARIMAX] Baseline results not found at {results_path}. "
              f"Continuing without ARIMAX comparison.")
        return None

    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        print(f"[ARIMAX] Could not read baseline results: {e}")
        return None

    # Columns come from train_arimax.py
    #   'Model', 'Out_Sample_RMSE', 'Directional_Accuracy', 'Correlation', ...
    rmse_col = 'Out_Sample_RMSE' if 'Out_Sample_RMSE' in df.columns else 'RMSE'

    baseline = {
        'model_name': str(df['Model'].iloc[0]) if 'Model' in df.columns else 'ARIMAX',
        'rmse': float(df[rmse_col].iloc[0]) if rmse_col in df.columns else None,
        'dir_acc': float(df['Directional_Accuracy'].iloc[0]) if 'Directional_Accuracy' in df.columns else None,
        'corr': float(df['Correlation'].iloc[0]) if 'Correlation' in df.columns else None,
        'sharpe': float(df['Sharpe_Ratio'].iloc[0]) if 'Sharpe_Ratio' in df.columns else None,
    }

    if all(v is not None for v in (baseline['rmse'], baseline['dir_acc'], baseline['corr'])):
        msg = (
            f"[ARIMAX] Loaded baseline ({baseline['model_name']}): "
            f"RMSE={baseline['rmse']:.4f}, "
            f"DA={baseline['dir_acc']:.3f}, "
            f"Corr={baseline['corr']:.4f}"
        )
        if baseline['sharpe'] is not None:
            msg += f", Sharpe={baseline['sharpe']:.4f}"
        print(msg)


    return baseline



def create_comprehensive_visualization(predictor, metrics, train_losses, val_losses):
    # Extract data
    predictions = metrics['Predictions']
    targets = metrics['Targets']
    rmse = metrics['RMSE']
    mae = metrics['MAE']
    dir_acc = metrics['Directional_Accuracy']
    correlation = metrics['Correlation']

    # Load ARIMAX baseline metrics from CSV (if available)
    arimax_baseline = load_arimax_baseline()
    if arimax_baseline is not None:
        arimax_rmse = arimax_baseline['rmse']
        arimax_dir_acc = arimax_baseline['dir_acc']
        arimax_corr = arimax_baseline['corr']
        arimax_label = arimax_baseline['model_name']
    else:
        arimax_rmse = None
        arimax_dir_acc = None
        arimax_corr = None
        arimax_label = 'ARIMAX'

    # Create subplot figure
    fig = plt.figure(figsize=(20, 24))

    # 1. Training Progress (Loss Curves)
    ax1 = plt.subplot(4, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    plt.title('LSTM Training Progress\nFeature Learning', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Early stopping annotation
    best_epoch = int(np.argmin(val_losses)) + 1
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
    plt.annotate(
        f'Early Stop\nEpoch {best_epoch}',
        xy=(best_epoch, min(val_losses)),
        xytext=(best_epoch + 5, min(val_losses) + 0.02),
        arrowprops=dict(arrowstyle='->', color='green', alpha=0.7)
    )

    # 2. Performance Comparison Bar Chart
    ax2 = plt.subplot(4, 2, 2)
    metrics_names = ['RMSE', 'Directional\nAccuracy (%)', 'Correlation\n(×100)']

    x = np.arange(len(metrics_names))
    width = 0.35

    lstm_values = [rmse, dir_acc * 100, correlation * 100]

    if arimax_rmse is not None and arimax_dir_acc is not None and arimax_corr is not None:
        arimax_values = [arimax_rmse, arimax_dir_acc * 100, arimax_corr * 100]

        bars1 = plt.bar(
            x - width / 2,
            arimax_values,
            width,
            label=f'{arimax_label} Baseline',
            color='lightcoral',
            alpha=0.8
        )
        bars2 = plt.bar(
            x + width / 2,
            lstm_values,
            width,
            label='LSTM',
            color='gold',
            alpha=0.8
        )
        title = 'LSTM vs ARIMAX Performance Comparison'
    else:
        bars1 = None
        bars2 = plt.bar(
            x,
            lstm_values,
            width,
            label='LSTM',
            color='gold',
            alpha=0.8
        )
        title = 'LSTM Performance (ARIMAX baseline not available)'

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Performance Score')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Value labels
    if bars1 is not None:
        for bar in bars1:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.1,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )

    for bar in bars2:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.1,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            color='darkred'
        )

    # 3. Time Series: Actual vs Predicted Returns
    ax3 = plt.subplot(4, 1, 2)

    time_index = range(len(predictions))

    plt.plot(time_index, targets, 'b-', label='Actual Returns', linewidth=1.5, alpha=0.8)
    plt.plot(time_index, predictions, 'r--', label='LSTM Predictions', linewidth=1.5, alpha=0.8)

    plt.title(
        f'LSTM Predictions vs Actual S&P 500 Returns\nDirectional Accuracy: {dir_acc:.1%}',
        fontsize=14,
        fontweight='bold'
    )
    plt.xlabel('Test Period (Trading Days)')
    plt.ylabel('Daily Returns (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Highlight correct directions
    correct_predictions = np.sign(targets) == np.sign(predictions)
    plt.fill_between(
        time_index,
        plt.ylim()[0],
        plt.ylim()[1],
        where=correct_predictions,
        alpha=0.1,
        color='green',
        label='Correct Direction'
    )

    # 4. Scatter Plot: Predicted vs Actual
    ax4 = plt.subplot(4, 2, 5)

    colors = [
        'green' if np.sign(targets[i]) == np.sign(predictions[i]) else 'red'
        for i in range(len(targets))
    ]

    plt.scatter(targets, predictions, c=colors, alpha=0.6, s=30)

    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)

    plt.title(
        f'Prediction Accuracy Scatter Plot\nCorrelation: {correlation:.4f}',
        fontsize=12,
        fontweight='bold'
    )
    plt.xlabel('Actual Returns (%)')
    plt.ylabel('Predicted Returns (%)')
    plt.grid(True, alpha=0.3)

    plt.text(
        0.05,
        0.95,
        f'R² = {correlation**2:.3f}\nCorrect Direction: {dir_acc:.1%}',
        transform=ax4.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        verticalalignment='top'
    )

    # 5. Residual Analysis
    ax5 = plt.subplot(4, 2, 6)
    residuals = targets - predictions

    plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(
        'Prediction Residuals Distribution',
        fontsize=12,
        fontweight='bold'
    )
    plt.xlabel('Prediction Error (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.axvline(
        residuals.mean(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {residuals.mean():.4f}%'
    )
    plt.legend()

    # 6. Improvement Metrics Visualization
    ax6 = plt.subplot(4, 2, 7)

    if arimax_rmse is not None and arimax_dir_acc is not None and arimax_corr is not None:
        improvements = [
            ('RMSE\nImprovement', ((arimax_rmse - rmse) / arimax_rmse) * 100),
            ('Directional\nImprovement', ((dir_acc - arimax_dir_acc) / arimax_dir_acc) * 100),
            ('Correlation\nGain', (correlation - arimax_corr) * 100),
        ]

        labels, values = zip(*improvements)
        colors_imp = ['lightgreen', 'gold', 'lightblue']

        bars = plt.bar(labels, values, color=colors_imp, alpha=0.8, edgecolor='black')
        plt.title(
            'LSTM Performance Improvements vs ARIMAX',
            fontsize=12,
            fontweight='bold'
        )
        plt.ylabel('Improvement (%)')
        plt.grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 1,
                f'+{height:.1f}%',
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=11
            )
    else:
        plt.axis('off')
        plt.text(
            0.5,
            0.5,
            'ARIMAX baseline metrics not available.\n'
            'Run train_arimax.py first to enable comparison.',
            ha='center',
            va='center',
            fontsize=11,
            fontweight='bold'
        )

    # 7. Directional Accuracy Over Time
    ax7 = plt.subplot(4, 2, 8)

    window = 50
    rolling_accuracy = []
    for i in range(window, len(targets)):
        subset_targets = targets[i - window:i]
        subset_predictions = predictions[i - window:i]
        accuracy = np.mean(np.sign(subset_targets) == np.sign(subset_predictions))
        rolling_accuracy.append(accuracy * 100)

    time_rolling = range(window, len(targets))
    plt.plot(time_rolling, rolling_accuracy, 'g-', linewidth=2, alpha=0.8)
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')

    if arimax_dir_acc is not None:
        plt.axhline(
            y=arimax_dir_acc * 100,
            color='orange',
            linestyle='--',
            alpha=0.7,
            label=f'{arimax_label} ({arimax_dir_acc*100:.1f}%)'
        )

    plt.title(
        f'Rolling Directional Accuracy ({window}-Day Window)',
        fontsize=12,
        fontweight='bold'
    )
    plt.xlabel('Test Period (Trading Days)')
    plt.ylabel('Directional Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(30, 90)

    plt.tight_layout()

    # Save the comprehensive plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'reports/figures/lstm_results_comprehensive_{timestamp}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comprehensive visualization saved: {filename}")

    plt.show()

    return fig


def visualize_lstm_results(predictor=None, metrics=None, train_losses=None, val_losses=None):
    print("Creating comprehensive LSTM results visualization...")

    fig1 = create_comprehensive_visualization(predictor, metrics, train_losses, val_losses)

    print("\nVISUALIZATION COMPLETE!")
    print("=" * 40)
    print("Created:")
    print("• Comprehensive analysis plots")
    print("• Summary dashboard")
    print("• Detailed results report")

    return fig1
