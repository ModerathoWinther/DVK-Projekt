import os

import plot

os.chdir('..')
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns


# ====================== CONFIG ======================
OUTPUT_DIR = 'results/analysis'
BACKTEST_DIR = 'results/backtest'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTROL = 'hold'

AGENT_FILES = {
    'price-only': f'{BACKTEST_DIR}/midas-price-only/midas-price-only.csv',
    'atr': f'{BACKTEST_DIR}/midas-atr/midas-atr.csv',
    'macd': f'{BACKTEST_DIR}/midas-macd/midas-macd.csv',
    'rsi': f'{BACKTEST_DIR}/midas-rsi/midas-rsi.csv',
    'atr-macd': f'{BACKTEST_DIR}/midas-atr-macd/midas-atr-macd.csv',
    'atr-rsi': f'{BACKTEST_DIR}/midas-atr-rsi/midas-atr-rsi.csv',
    'macd-rsi': f'{BACKTEST_DIR}/midas-macd-rsi/midas-macd-rsi.csv',
    'all-indicators': f'{BACKTEST_DIR}/midas-all-indicators/midas-all-indicators.csv',
    'hold': f'{BACKTEST_DIR}/midas-hold/midas-hold.csv',
}

METRICS = ['win_rate', 'profit_factor', 'expectancy', 'max_drawdown', 'sharpe_ratio']


def load_agent(filepath: str, agent_name: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df['agent'] = agent_name

    # Handle infinite profit factors
    df['profit_factor'] = df['profit_factor'].replace([np.inf, -np.inf, float('inf')], 100.0)

    # Calcula
    if 'current_equity' in df.columns:
        d = (df['current_equity'] - 100000) / 100000 * 100
    elif 'equity' in df.columns:
        d = (df['equity'] - 100000) / 100000 * 100

    n_before = len(df)
    models_df = df.copy()


    print(f"Loaded {agent_name}: {len(models_df)} valid episodes")
    return models_df


def load_all_data() -> dict:
    data = {}
    for name, path in AGENT_FILES.items():
        if os.path.exists(path):
            data[name] = load_agent(path, name)
            print(f"Loaded {name}: {len(data[name])} episodes")
        else:
            print(f"Warning: Missing file for {name}")

    return data


# ====================== STATISTICS ======================
def run_omnibus(data: dict, metric: str):
    """Safely run omnibus test with proper checks."""
    groups = []
    group_names = []

    for agent, df in data.items():
        vals = df[metric].dropna().values
        if len(vals) >= 3:  # minimum for meaningful test
            groups.append(vals)
            group_names.append(agent)
        else:
            print(f"  Warning: {agent} has only {len(vals)} valid {metric} values → skipped in omnibus")

    if len(groups) < 2:
        print(f"  ERROR: Only {len(groups)} valid groups for {metric}. Cannot run omnibus test.")
        return np.nan, np.nan, 'Insufficient data'

    # Check normality
    normality_pass = all(shapiro(g)[1] > 0.05 for g in groups)

    if normality_pass:
        try:
            stat, p = stats.f_oneway(*groups)
            test_name = 'One-way ANOVA'
        except Exception as e:
            print(f"  ANOVA failed: {e}")
            stat, p = np.nan, np.nan
            test_name = 'One-way ANOVA (failed)'
    else:
        stat, p = kruskal(*groups)
        test_name = 'Kruskal-Wallis'

    print(f"  {test_name} on {len(groups)} groups: stat={stat:.3f}, p={p:.4f}")
    return stat, p, test_name


def test_vs_control(data: dict, metric: str, control_name=CONTROL):
    control_vals = data[control_name][metric].dropna().values
    n_treat = len(data) - 1
    rows = []

    for agent, df in data.items():
        if agent == control_name:
            continue
        vals = df[metric].dropna().values

        # Normality check
        _, p_a = shapiro(vals) if len(vals) >= 3 else (0, 1)
        _, p_c = shapiro(control_vals) if len(control_vals) >= 3 else (0, 1)
        both_normal = p_a > 0.05 and p_c > 0.05

        if both_normal:
            stat, p_raw = stats.ttest_ind(vals, control_vals, equal_var=False)
            test_name = 'Welch t-test'
        else:
            stat, p_raw = mannwhitneyu(vals, control_vals, alternative='two-sided')
            test_name = 'Mann-Whitney U'

        p_adj = min(p_raw * n_treat, 1.0)

        pooled = np.sqrt((np.std(vals, ddof=1) ** 2 + np.std(control_vals, ddof=1) ** 2) / 2)
        cohens_d = (np.mean(vals) - np.mean(control_vals)) / (pooled + 1e-8)

        rows.append({
            'agent': agent,
            'n': len(vals),
            'mean': np.mean(vals),
            'median': np.median(vals),
            'std': np.std(vals, ddof=1),
            'control_mean': np.mean(control_vals),
            'diff': np.mean(vals) - np.mean(control_vals),
            'test': test_name,
            'stat': stat,
            'p_raw': p_raw,
            'p_adj': p_adj,
            'cohens_d': cohens_d,
            'sig': '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'
        })

    return pd.DataFrame(rows).sort_values('mean', ascending=False)

def validate_data(data: dict):
    print("\n=== DATA VALIDATION ===")
    for agent, df in data.items():
        print(f"{agent:20} | rows={len(df):3d} | ", end='')
        for m in METRICS:
            if m in df.columns:
                valid = df[m].dropna().count()
                print(f"{m[:4]}:{valid} ", end='')
        print()


def plot_boxplots(data):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Define which metrics should display the B&H line
    benchmarked_metrics = ['sharpe_ratio']

    for i, metric in enumerate(METRICS[:6]):
        ax = axes[i]

        # 1. Prepare data and filter out 'hold' for the boxes
        full_plot_df = pd.concat([df.assign(agent=name) for name, df in data.items()])
        box_plot_df = full_plot_df[full_plot_df['agent'] != 'hold']

        sns.boxplot(data=box_plot_df, x='agent', y=metric, ax=ax, palette="Set2")
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        # 2. Only draw the B&H line for Sharpe and Return %
        if metric in benchmarked_metrics and 'hold' in data:
            control_mean = data['hold'][metric].mean()
            ax.axhline(control_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8, label='Buy&Hold Mean')
            ax.legend()
        else:
            # For win_rate, profit_factor, etc., we just show the model distributions
            pass

    # Hide the 6th subplot if you only have 5 metrics
    if len(METRICS) < 6:
        axes[-1].set_visible(False)

    plt.suptitle('Trained Model Performance Distributions\n(Benchmark line shown for Risk-Adjusted Returns only)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/boxplots_all_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/boxplots_all_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Boxplots generated. B&H comparison limited to: {benchmarked_metrics}")


def plot_sharpe_bar(data_source):
    means = {name: df['sharpe_ratio'].mean() for name, df in data_source.items()}
    stds = {name: df['sharpe_ratio'].std() for name, df in data_source.items()}

    df_plot = pd.DataFrame({'agent': list(means.keys()),
                            'sharpe': list(means.values()),
                            'std': list(stds.values())})
    df_plot = df_plot.sort_values('sharpe', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='agent', y='sharpe', palette="Blues_d")
    plt.errorbar(x=range(len(df_plot)), y=df_plot['sharpe'], yerr=df_plot['std'],
                 fmt='none', c='black', capsize=5)

    plt.title('Mean Sharpe Ratio by Agent (± Std)', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sharpe_ratio_bar.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Sharpe bar plot saved")


def plot_ablation_heatmap(data):
    agents = [a for a in data.keys() if a != 'hold']
    metrics = ['sharpe_ratio', 'win_rate', 'expectancy']

    matrix = pd.DataFrame({m: {a: data[a][m].mean() for a in agents} for m in metrics})
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix.T, annot=True, cmap='RdYlGn', fmt='.3f', linewidths=0.5, cbar_kws={'label': 'Mean Value'})
    plt.title('Indicator Ablation Heatmap - Mean Performance')
    plt.ylabel('Metric')
    plt.xlabel('Agent')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ablation_heatmap.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Ablation heatmap saved")

def plot_mismatched_data(gold_df, model_results):

    plt.figure(figsize=(12, 6))
    gold_initial = gold_df['close'].iloc[0]
    gold_pct = (gold_df['close'] - gold_initial) / gold_initial * 100
    plt.plot(gold_df['date'], gold_pct, label='Gold Price', color='gold', alpha=0.4, linewidth=1)

    # 2. Plot Model Equity (The 50 datapoints)
    for name, df in model_results.items():
        initial_eq = df['current_equity'].iloc[0]
        equity_pct = (df['current_equity'] - initial_eq) / initial_eq * 100

        # Use drawstyle='steps-post' to show the equity jumps between episodes
        plt.plot(df['date'], equity_pct, label=name, drawstyle='steps-post', linewidth=2)

    plt.title("Cumulative Return: 50-Episode Portfolio vs. Underlying Gold")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()


# ====================== MAIN ======================
if __name__ == "__main__":
    data = load_all_data()
    print(data)
    validate_data(data)
    # plot_boxplots(data)
    plot_sharpe_bar(data)
    # plot_ablation_heatmap(data)
    prices = pd.read_csv('data/raw/test/raw_test.csv')
    # Remove agents with zero rows
    data = {k: v for k, v in data.items() if len(v) > 0}

    if CONTROL not in data:
        print(f"CRITICAL ERROR: Control '{CONTROL}' not loaded!")

    summary = []
    for agent, df in data.items():
        row = {'agent': agent, 'n_episodes': len(df)}
        for m in METRICS:
            if m in df.columns:
                row[f'{m}_mean'] = df[m].mean()
                row[f'{m}_std'] = df[m].std(ddof=1)
                row[f'{m}_median'] = df[m].median()
        summary.append(row)
    summary_df = pd.DataFrame(summary).sort_values('expectancy_mean', ascending=False)
    summary_df.to_csv(f'{OUTPUT_DIR}/summary_all_agents.csv', index=False)
    print("\n=== SUMMARY TABLE ===")
    print(summary_df.round(4))


    # Per-metric analysis
    for metric in METRICS:
        if metric not in list(data.values())[0].columns:
            continue

        print(f"\n{'=' * 80}")
        print(f"  {metric.upper().replace('_', ' ')}")
        print('=' * 80)

        stat, p, test_name = run_omnibus(data, metric)
        print(f"{test_name}: stat={stat:.3f}, p={p:.4f}")

        if p > 0.05:
            print("→ No overall significant differences.")
            continue

        result_df = test_vs_control(data, metric)
        result_df.to_csv(f'{OUTPUT_DIR}/{metric}_vs_buyhold.csv', index=False)

        # Print nice table
        print(result_df[['agent', 'mean', 'median', 'diff', 'p_adj', 'cohens_d', 'sig']].round(4))

    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}/")
    backtest_sharpe = {
        'Price-Only': 1.411,
        'ATR-MACD': 1.394,
        'RSI': 1.286,
        'ATR-RSI': 1.191,
        'All Indicators': 1.169,
        'ATR': 1.093,
        'MACD-RSI': 0.192,
        'MACD': 0.060,
        'Buy-and-Hold': 0.304,
    }

    live_profit = {
        'ATR': 1838.0,
        'Buy-and-Hold': 1363.7,
        'MACD-RSI': 385.6,
        'ATR-RSI': 171.5,
        'All Indicators': 95.9,
        'ATR-MACD': -36.4,
        'MACD': -106.9,
        'Price-Only': -619.7,
        'RSI': -1496.9,
    }

    agents = list(backtest_sharpe.keys())
    bt_ranks = pd.Series(backtest_sharpe).rank(ascending=False)
    lt_ranks = pd.Series(live_profit).rank(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agents))
    agents_bt_sorted = bt_ranks.sort_values().index.tolist()

    for agent in agents_bt_sorted:
        bt_r = bt_ranks[agent]
        lt_r = lt_ranks[agent]
        color = '#d62728' if agent == 'RSI' else \
            '#2ca02c' if agent == 'ATR' else '#aec7e8'
        ax.plot([0, 1], [bt_r, lt_r], 'o-', color=color, linewidth=2,
                markersize=8, label=agent)
        ax.text(-0.05, bt_r, agent, ha='right', va='center', fontsize=9)
        ax.text(1.05, lt_r, agent, ha='left', va='center', fontsize=9)

    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Backtest\n(Sharpe rank)', 'Live Test\n(Profit rank)'], fontsize=11)
    ax.invert_yaxis()
    ax.set_ylabel('Rank (1 = best)')
    ax.set_title('Agent Performance Rank: Backtest vs Live Test')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig('results/rank_comparison.pdf', dpi=150)
    # Then proceed with summary + per-metric analysis...