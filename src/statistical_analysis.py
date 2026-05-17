import os
from typing import Any

from pandas import DataFrame

os.chdir('..')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, f_oneway, kruskal
import scikit_posthocs as sp


EVAL_RESULTS_DIR = 'eval-results'
BACKTEST_DIR = os.path.join(EVAL_RESULTS_DIR, 'backtest')
LIVE_TEST_DIR = os.path.join(EVAL_RESULTS_DIR, 'live_test')
OUTPUT_DIR = 'results/analysis'


class StatisticalAnalysis:
    def __init__(self, params):
        with open('statistical_analysis.yml', 'r') as file:
            param_list = yaml.safe_load(file)
            self.params = param_list[params]
        self.backtest_results = {}
        self.models = self.params['models']
        self.metrics = self.params['metrics']
        self._load_episode_results(self.models)

        print(f'Metrics analyzed: {self.metrics}')
        print(f'Models: {[model for model in self.backtest_results]}')

        self._full_analysis()

    def _load_episode_results(self, models) -> None:
        for model in models:
            with open(f'{BACKTEST_DIR}/{model}/{model}.csv') as f:
                self.backtest_results[model] = pd.read_csv(f)

    def _check_normality(self, metric: str) -> pd.DataFrame:
        rows = []
        for agent, df in self.backtest_results.items():
            values = df[metric].dropna().values
            if len(values) < 3:
                continue
            stat, p = shapiro(values)
            rows.append({
                'agent': agent,
                'n': len(values),
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'shapiro_stat': stat,
                'shapiro_p': p,
                'normal': p > 0.05
            })
        return pd.DataFrame(rows)

    def _run_omnibus_test(self, metric: str) -> tuple[float, float, str]:
        groups = [df[metric].dropna().values for df in self.backtest_results.values()]
        normality = all(shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)

        if normality:
            stat, p = f_oneway(*groups)
            return stat, p, 'One-way ANOVA'
        else:
            stat, p = kruskal(*groups)
            return stat, p, 'Kruskal-Wallis'

    def _run_dunnett(self, metric: str, control: str = 'buy-hold') -> pd.DataFrame:

        control_vals = self.backtest_results[control][metric].dropna().values
        treatment_agents = [name for name in self.backtest_results if name != control]

        rows = []
        for agent in treatment_agents:
            agent_vals = self.backtest_results[agent][metric].dropna().values

            _, p_normal_agent = shapiro(agent_vals)
            _, p_normal_control = shapiro(control_vals)
            both_normal = p_normal_agent > 0.05 and p_normal_control > 0.05

            if both_normal:
                t_stat, p_raw = stats.ttest_ind(agent_vals, control_vals,
                                                equal_var=False)
                test = 't-test (Welch)'
            else:
                t_stat, p_raw = stats.mannwhitneyu(agent_vals, control_vals,
                                                   alternative='two-sided')
                test = 'Mann-Whitney U'

            n_treatments = len(treatment_agents)
            p_corrected = min(p_raw * n_treatments, 1.0)

            pooled_std = np.sqrt((np.std(agent_vals, ddof=1) ** 2 +
                                  np.std(control_vals, ddof=1) ** 2) / 2)
            cohens_d = (np.mean(agent_vals) - np.mean(control_vals)) / (pooled_std + 1e-10)

            rows.append({
                'agent': agent,
                'mean': np.mean(agent_vals),
                'std': np.std(agent_vals, ddof=1),
                'control_mean': np.mean(control_vals),
                'diff': np.mean(agent_vals) - np.mean(control_vals),
                'test': test,
                'statistic': t_stat,
                'p_raw': p_raw,
                'p_corrected': p_corrected,
                'cohens_d': cohens_d,
                'significant': p_corrected < 0.05,
            })

        return pd.DataFrame(rows).sort_values('mean', ascending=False)

    def _full_analysis(self) -> dict[Any, DataFrame]:
        all_results = {}

        for metric in self.metrics:
            print(f"\n{'='*65}")
            print(f"  {metric.upper()}")
            print('='*65)

            norm_df = self._check_normality(metric)
            n_normal = norm_df['normal'].sum()
            print(f"  Normality: {n_normal}/{len(norm_df)} groups pass Shapiro-Wilk")

            stat, p_omni, test_name = self._run_omnibus_test(metric)
            print(f"  {test_name}: F/H={stat:.3f}, p={p_omni:.4f}")

            if p_omni > 0.05:
                print(f"  → No significant group differences. Skipping post-hoc.")
                continue

            results_df = self._run_dunnett(metric)
            all_results[metric] = results_df

            print(f"\n  {'Agent':<22} {'Mean':>8} {'vs BH':>8} "
                  f"{'Stat':>8} {'p(raw)':>8} {'p(adj)':>8} {'d':>7} {'Sig':>5}")
            print('  ' + '-'*63)
            for _, row in results_df.iterrows():
                sig = '***' if row['p_corrected'] < 0.001 else \
                      '**'  if row['p_corrected'] < 0.01  else \
                      '*'   if row['p_corrected'] < 0.05  else ''
                print(f"  {row['agent']:<22} "
                      f"{row['mean']:>8.3f} "
                      f"{row['diff']:>+8.3f} "
                      f"{row['statistic']:>8.3f} "
                      f"{row['p_raw']:>8.4f} "
                      f"{row['p_corrected']:>8.4f} "
                      f"{row['cohens_d']:>7.3f} "
                      f"{sig:>5}")

            results_df.to_csv(f"{OUTPUT_DIR}/{metric}_dunnett.csv", index=False)

        return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical analysis: name model to analyze.')
    parser.add_argument('analysis_params', help='Name parameter set for models and metrics to analyze in statistical_analysis.yml.')
    args = parser.parse_args()

    analysis_params = args.analysis_params
    model_analysis = StatisticalAnalysis(analysis_params)



