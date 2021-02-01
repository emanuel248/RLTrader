import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from datetime import datetime
import time
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
from plotly.subplots import make_subplots

# Instantiate the parser
parser = argparse.ArgumentParser(description='Display study statistics')
parser.add_argument('study', type=str,
                    help='Name for study')

args = parser.parse_args()


if __name__ == '__main__':
    study = optuna.load_study(study_name=args.study, storage='sqlite:///params_ppo2.db')
    plot_parallel_coordinate(study).show()
    #plot_param_importances(study).show()
