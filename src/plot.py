import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scikit_posthocs as sp
import seaborn as sns

def plot_metric_comparison(data: dict, metric: str,
                            dunnett_df: pd.DataFrame,
                           output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    agent_order = dunnett_df.sort_values('mean', ascending=False)['agent'].tolist()
    agent_order = ['buy-hold'] + agent_order

    plot_data = [data[a][metric].dropna().values for a in agent_order]
    positions = range(len(agent_order))

    bp = ax.boxplot(plot_data, positions=positions, patch_artist=True,
                    notch=False, showfliers=True)

    for i, (patch, name) in enumerate(zip(bp['boxes'], agent_order)):
        patch.set_facecolor('#2196F3' if name == 'buy-hold' else '#4CAF50')
        patch.set_alpha(0.7)

    y_max = max(np.percentile(d, 95) for d in plot_data) * 1.1
    for i, agent in enumerate(agent_order[1:], start=1):
        row = dunnett_df[dunnett_df['agent'] == agent]
        if row.empty:
            continue
        p = row['p_corrected'].values[0]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(i, y_max, sig, ha='center', va='bottom', fontsize=11)

    ax.set_xticks(positions)
    ax.set_xticklabels(agent_order, rotation=30, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_"," ").title()} — All Agents vs Buy-and-Hold')
    ax.axhline(y=np.mean(data['buy-hold'][metric]), color='blue',
               linestyle='--', alpha=0.5, label='Buy-hold mean')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_boxplot.pdf", format='pdf')
    plt.close()


def plot_indicator_ablation_heatmap(all_results: dict, output_dir: str, agent_list: list) -> None:

    agents = [a for a in agent_list if a != 'buy-hold']
    metrics_to_show = ['sharpe_ratio', 'expectancy', 'profit_factor', 'win_rate']

    matrix = pd.DataFrame(index=agents, columns=metrics_to_show)
    for metric in metrics_to_show:
        if metric in all_results:
            for _, row in all_results[metric].iterrows():
                if row['agent'] in agents:
                    matrix.loc[row['agent'], metric] = row['mean']

    matrix = matrix.astype(float)

    normalised = (matrix - matrix.mean()) / matrix.std()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(normalised.values, cmap='RdYlGn', aspect='auto',
                   vmin=-2, vmax=2)

    ax.set_xticks(range(len(metrics_to_show)))
    ax.set_yticks(range(len(agents)))
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics_to_show])
    ax.set_yticklabels(agents)

    for i in range(len(agents)):
        for j in range(len(metrics_to_show)):
            val = matrix.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color='black')

    plt.colorbar(im, ax=ax, label='Z-score (within metric)')
    ax.set_title('Indicator Ablation — Mean Performance per Agent\n(colour = z-score within metric)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_heatmap.pdf", format='pdf')
    plt.close()

def save_box_plot(data):
    metrics = ['sharpe_ratio', 'win_rate', 'return_pct', 'profit_factor', 'expectancy']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.boxplot(data=data, x='agent', y=metric, ax=axes[i], palette="Set2")
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.suptitle('Performance Distribution Across Agents (50 Episodes)', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/analysis/boxplots.pdf', dpi=300, bbox_inches='tight')
    plt.show()