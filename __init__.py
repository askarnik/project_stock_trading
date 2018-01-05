import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lib import ta
from lib.plr import PLR, connect_line, flatten, trading_signal, up_down_trend, calc_profit, best_epsilon
from lib.modeling import deskew_df, test_model, get_cm, tts_stock

import inspect
import string
import os

import pickle

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR