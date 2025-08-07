import pandas as pd 
import numpy as np  
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from autogluon.tabular import TabularPredictor
import sklearn.metrics
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from ForestDiffusion import ForestDiffusionModel as ForestFlowModel
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator

import warnings
warnings.filterwarnings(action='ignore')



params = {
    "n_estimators": 400,
    "max_leaf_nodes": 15000,
    "n_jobs": -1,
    "random_state": 0,
    "bootstrap": True,
    "criterion": "entropy"
}
rf = RandomForestClassifier(**params)

train_df = pd.read_csv('./DS/original_train.csv')
test_df = pd.read_csv('./DS/original_test.csv')

X_train = train_df.drop('Label', axis=1)
y_train = train_df['Label']

X_test = test_df.drop('Label', axis=1)
y_test = test_df['Label']

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
rf.fit(X_train_scaled, y_train)     
y_preds = rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_preds)
print(f'Accuracy of model trained with orginal training set: {accuracy}')
X, y = GANGenerator(gen_x_times=1.1, cat_cols=None,
    bot_filter_quantile=0.001, top_filter_quantile=0.999, is_post_process=True,
    adversarial_model_params={
        "metrics": "AUC", "max_depth": 10, "max_bin": 100, 
        "learning_rate": 0.09, "random_state": 42, "n_estimators": 500,'verbose_eval':False,'verbose': -1,'silent':True,
    }, pregeneration_frac=2, only_generated_data=True,
    gen_params = {"batch_size": 1000, "patience": 25, "epochs" : 500,}).generate_data_pipe(pd.DataFrame(X_train_scaled), pd.DataFrame(y_train), pd.DataFrame(X_test_scaled), deep_copy=True, only_adversarial=False, use_adversarial=True)
Xy = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
df = pd.DataFrame(Xy, columns=[cname for cname in train_df.columns])
rf.fit(df.drop('Label', axis=1), df['Label'])
y_preds = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_preds)
df.to_csv(f'./dataset/TGAN_dataset.csv', index=False)
print(f'Accuracy of model trained with TabGAN training set: {accuracy}') 