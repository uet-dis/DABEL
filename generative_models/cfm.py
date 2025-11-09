import pandas as pd
import numpy as np
import joblib
# Plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import os

# Data processing, metrics and modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, roc_auc_score, classification_report, roc_auc_score, auc

from imblearn.over_sampling import SMOTE,SVMSMOTE
from collections import Counter

# Machine Learning Libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.models import model_from_json
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
import copy
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import MinMaxScaler

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher



warnings.filterwarnings("ignore")
# Display Settings

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 25)

train = pd.read_csv('./DS/original_train.csv')
test = pd.read_csv('./DS/original_test.csv')

train.to_csv('./DS/train_initial.csv', index=False)
test.to_csv('./DS/test_initial.csv', index=False)

X_train_df = train.drop(['Label'], axis=1)
y_train_df = train['Label']
X_test_df = test.drop(['Label'], axis=1)
y_test_df = test['Label']

X_train = X_train_df.to_numpy()
X_test = X_test_df.to_numpy()

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_df)
np.save('./DS/y_train_encoded.npy', y_train)

y_test = label_encoder.fit_transform(y_test_df)
np.save('./DS/y_test_encoded.npy', y_test)

# set seed
seed = 1980
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# Main hyperparameters
n_t = 50  # number of flow steps (higher is better, 50 is enough for great performance)
duplicate_K = 10  # number of different noise sample per real data sample (higher is better)

# XGBoost hyperparameters
max_depth = 7
n_estimators = 100
eta = 0.3
tree_method = "hist"
reg_lambda = 0.0
reg_alpha = 0.0
subsample = 1.0
Xy_fake = None
y_uniques = None
regr = None
c = None

# Function used for training one model
# Regressor
def train_parallel(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        objective="reg:squarederror",
        eta=eta,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        seed=666,
        tree_method=tree_method,
        device="cpu",
    )

    y_no_miss = ~np.isnan(y_train)
    model.fit(X_train[y_no_miss, :], y_train[y_no_miss])

    return model
def my_model(t, xt, mask_y=None):
    # xt is [b*c]
    xt = xt.reshape(xt.shape[0] // c, c)  # [b, c]

    # Output from the models
    out = np.zeros(xt.shape)  # [b, c]
    i = int(round(t * (n_t - 1)))
    for j, label in enumerate(y_uniques):
        for k in range(c):
            out[mask_y[label], k] = regr[j][i][k].predict(xt[mask_y[label], :])

    out = out.reshape(-1)  # [b*c]
    return out

# Simple Euler ODE solver (nothing fancy)
def euler_solve(x0, my_model, N=100):
    h = 1 / (N - 1)
    x_fake = x0
    t = 0
    # from t=0 to t=1
    for i in range(N - 1):
        x_fake = x_fake + h * my_model(t=t, xt=x_fake)
        t = t + h
    return x_fake    
def testSamples(Xy, X_test, y_test):    
    params = {
        'gpu_id': 0,
        'tree_method': 'hist',
        'max_depth': 128,
        'objective': 'binary:logistic', # 'multi:softmax',
        'booster': 'gbtree',
        'learning_rate': 0.1,
        'eval_metric': 'auc'
    }
    XGB = xgb.XGBClassifier(**params)    
    df = pd.DataFrame(Xy, columns=[cname for cname in train.columns])
    XGB.fit(df.drop('Label', axis=1), df['Label'])
    y_preds = XGB.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    print(accuracy)
    return accuracy     

def generateCFM(batch_size = 2000):
    try:
        train = pd.read_csv('./DS/original_train.csv')
        test = pd.read_csv('./DS/original_test.csv')
        
        X_train_df = train.drop(['Label'], axis=1)
        y_train_df = train['Label']
        X_test_df = test.drop(['Label'], axis=1)
        y_test_df = test['Label']
    
        X_train = X_train_df.to_numpy()
        X_test = X_test_df.to_numpy()
        
        label_encoder = LabelEncoder()
        
        y_train = label_encoder.fit_transform(y_train_df)
        y_test = label_encoder.fit_transform(y_test_df)
    
        X = X_train
        y = y_train
        # print(X)
        # print(y)
    except Exception as e:
        print("Error loading or processing data:", e)
    # shuffle the observations
    new_perm = np.random.permutation(X.shape[0])
    np.take(X, new_perm, axis=0, out=X)
    np.take(y, new_perm, axis=0, out=y)

    # Save data before adding missing values
    X_true, y_true = copy.deepcopy(X), copy.deepcopy(y)
    Xy_true = np.concatenate((X_true, np.expand_dims(y_true, axis=1)), axis=1)

    # Save shape
    b, c = X.shape

    # we duplicate the data multiple times, so that X0 is k times bigger, so that we can have k random noise z associated per sample
    X1 = np.tile(X, (duplicate_K, 1))

    # Generate noise data
    X0 = np.random.normal(size=X1.shape)

    # Saving the freqency of the classes and storing label masks for later
    y_uniques, y_probs = np.unique(y, return_counts=True)
    y_probs = y_probs / np.sum(y_probs)
    mask_y = {}  # mask for which observations has a specific value of y
    for i in range(len(y_uniques)):
        mask_y[y_uniques[i]] = np.zeros(b, dtype=bool)
        mask_y[y_uniques[i]][y == y_uniques[i]] = True
        mask_y[y_uniques[i]] = np.tile(mask_y[y_uniques[i]], (duplicate_K))

    # for j in y_uniques:
    #     print("Checking mask_y[{}] size: {}, expected size: {}".format(j, mask_y[j].shape, b * duplicate_K))
    
    n_y = len(y_uniques)  # number of classes

    # Build [X(t), y] at multiple values of t

    # Define Independent Conditional Flow Matching (I-CFM)
    FM = ConditionalFlowMatcher(sigma=0.0)

    # Time levels
    t_levels = np.linspace(1e-3, 1, num=n_t)

    # Interpolation between x0 and x1 (xt)
    X_train = np.zeros((n_t, X0.shape[0], X0.shape[1]))  # [n_t, b, c]

    # Output to predict (ut)
    y_train = np.zeros((n_t, X0.shape[0], X0.shape[1]))  # [n_t, b, c]

    print("start")
    # Fill with xt and ut
    for i in range(n_t):
        t = torch.ones(X0.shape[0]) * t_levels[i]  # current t
        _, xt, ut = FM.sample_location_and_conditional_flow(
            torch.from_numpy(X0), torch.from_numpy(X1), t=t
        )
        X_train[i], y_train[i] = xt.numpy(), ut.numpy()

    print("start1")
    # print("X_train shape:", X_train.shape)
    # print(y_train)
    # %%time
    # Train all model(s); fast if you have a decent multi-core CPU, but extremely slow on Google Colab because it uses a weak 2-core CPU
    regr = Parallel(n_jobs=-1, max_nbytes=None, backend="threading")(  # using all cpus
        delayed(train_parallel)(
            X_train.reshape(n_t, b * duplicate_K, c)[i][mask_y[j], :],
            y_train.reshape(n_t, b * duplicate_K, c)[i][mask_y[j], k],
            # y_train[mask_y[j]]
        )
        for i in range(n_t)
        for j in y_uniques
        for k in range(c)
    )

    # Replace fits with doubly loops to make things easier
    regr_ = [[[None for k in range(c)] for i in range(n_t)] for j in y_uniques]
    current_i = 0
    for i in range(n_t):
        for j in range(len(y_uniques)):
            for k in range(c):
                regr_[j][i][k] = regr[current_i]
                current_i += 1
    regr = regr_    

        
    # Testing with XGBoost
    params = {
        'device': 'cuda',
        'max_depth': 128,
        'objective': 'binary:logistic', # 'multi:softmax',
        'booster': 'gbtree',
        'learning_rate': 0.1,
        'eval_metric': 'auc'
    }
    
    print("start2")
    try:
        XGB = xgb.XGBClassifier(**params)
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_df)
        y_test_encoded = label_encoder.transform(y_test_df)
        XGB.fit(X_train_df, y_train_encoded)
        y_preds = XGB.predict(X_test)
        accuracy = accuracy_score(y_test, y_preds)
        print(f'Original accuracy: {accuracy}')
        joblib.dump(XGB, './DS/xgb_model_final.pkl')
    except Exception as e:
        print("Failed to fit the model:", e)
        np.save('./DS/X_train_current_state.npy', X_train)
        np.save('./DS/y_train_current_state.npy', y_train)
    
    LOOP = 5
    new_acc = accuracy
    MAX = new_acc
    i = 0
    # while (new_acc <= accuracy + 0.02) :
    i = i+1
    # Generate prior noise
    x0 = np.random.normal(size=(batch_size, c))
    # x0 = batch_size / 2
    # print(x0.shape)

    # Generate random labels for the outcome
    label_y_fake = y_uniques[np.argmax(np.random.multinomial(1, y_probs, size=x0.shape[0]), axis=1)]
    mask_y_fake = {}  # mask for which observations has a specific value of y
    for i in range(len(y_uniques)):
        mask_y_fake[y_uniques[i]] = np.zeros(x0.shape[0], dtype=bool)
        mask_y_fake[y_uniques[i]][label_y_fake == y_uniques[i]] = True
    # print(label_y_fake.shape)
    # ODE solve
    ode_solved = euler_solve(my_model=partial(my_model, mask_y=mask_y_fake), x0=x0.reshape(-1), N=n_t)  
    # [t, b*c]
    solution = ode_solved.reshape(batch_size, c)  # [b, c]

    # # invert the min-max normalization
    # solution = scaler.inverse_transform(solution)

    # # clip to min/max values
    # small = (solution < X_min).astype(float)
    # solution = small * X_min + (1 - small) * solution
    # big = (solution > X_max).astype(float)
    # solution = big * X_max + (1 - big) * solution

    # Concatenate the y label
    Xy_fake = np.concatenate((solution, np.expand_dims(label_y_fake, axis=1)), axis=1)

    new_acc = testSamples(Xy_fake, X_test, y_test)
    if (new_acc > MAX):
        MAX = new_acc
        df = pd.DataFrame(Xy_fake, columns=[cname for cname in train.columns])
        df.to_csv(f'./DS/CFM_dataset.csv', index=False)
        print(f'Save generated samples with accuracy of {new_acc}')
    # if (i>LOOP):
    #     break

generateCFM()

print(Xy_fake.shape)

params = {
    'device': 'cuda',
    'max_depth': 128,
    'objective': 'binary:logistic', # 'multi:softmax',
    'booster': 'gbtree',
    'learning_rate': 0.1,
    'eval_metric': 'auc'
}
XGB = xgb.XGBClassifier(**params)

XGB.fit(X_train_df, y_train_df)
y_preds = XGB.predict(X_test)
accuracy = accuracy_score(y_test, y_preds)
print(accuracy)
print(y_train_df.value_counts())

df = pd.DataFrame(Xy_fake, columns=[cname for cname in train.columns])
XGB.fit(df.drop('Label', axis=1), df['Label'])
y_preds = XGB.predict(X_test)
accuracy = accuracy_score(y_test, y_preds)
print(accuracy)
print(df['Label'].value_counts())
