import os
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
            self.models = param_list[params]
        self.backtest_results = {}
        self._load_episode_results(self.models)

    def _load_episode_results(self, models) -> None:
        for model in models:
            with open(f'{BACKTEST_DIR}/{model}/{model}.csv') as f:
                self.backtest_results[model] = pd.read_csv(f)




def run(models):
    print(f'Analyzing model: {models}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical analysis: name model to analyze.')
    parser.add_argument('model_name', help='Name the parameter name with a set of models defined in statistical_analysis.yml.')
    args = parser.parse_args()

    model_name = args.model_name

    model_analysis = StatisticalAnalysis(model_name)

