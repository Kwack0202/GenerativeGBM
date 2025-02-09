# ======================================================
# Origin data download
import FinanceDataReader as fdr
import pandas_datareader.data as pdr

# ======================================================
# TA-Lib
import talib

# ======================================================
# basic library
import argparse
import warnings

import pandas as pd
import numpy as np

import time
import math
import os
import os.path
import random
import shutil
import glob

from tqdm import tqdm
from datetime import datetime

# ======================================================
# visualize
import matplotlib.pyplot as plt

# ======================================================
# Managing files
import openpyxl
import pickle

# ======================================================
# Data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ======================================================
# torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm

# ======================================================
# gmm preprocessing
from scipy.optimize import fmin
from scipy.special import lambertw
from scipy.stats import kurtosis, norm
from scipy.stats import ks_2samp
import statsmodels.api as sm
