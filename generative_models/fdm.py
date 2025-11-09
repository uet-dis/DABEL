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
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import shutil
from ForestDiffusion import ForestDiffusionModel as ForestFlowModel

import warnings
warnings.filterwarnings(action='ignore')

# Generate using Forest Diffusion Model
def generateFDM(model):
    train = pd.read_csv('./DS/original_train.csv')
    test = pd.read_csv('./DS/original_test.csv')
    
    X_train = train.drop(['Label'], axis=1)
    y_train = train['Label']
    X_test = test.drop(['Label'], axis=1)
    y_test = test['Label']
    
    label_encoder = LabelEncoder()
    
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    forest_model = ForestFlowModel(
        X_train,
        label_y=y_train,
        n_t=50,
        duplicate_K=100,
        bin_indexes=[],
        cat_indexes=[],
        int_indexes=[],
        diffusion_type="flow",
        n_jobs=-1,
        seed=1,
    )    
    model.fit(X_train, y_train)     
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    print(f'Accuracy of model trained with orginal training set: {accuracy}')

    Xy = forest_model.generate(batch_size=2000)
    df = pd.DataFrame(Xy, columns=[cname for cname in train.columns])

    model.fit(df.drop('Label', axis=1), df['Label'])
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    df.to_csv(f'./dataset/FDM_dataset.csv', index=False)
    print(f'Accuracy of model trained with FDM training set: {accuracy}')

params = {
    'gpu_id': 0,
    'tree_method': 'hist',
    'max_depth': 128,
    'objective': 'binary:logistic', # 'multi:softmax',
    'booster': 'gbtree',
    'learning_rate': 0.1,
    'eval_metric': 'auc'
}
XGB = XGBClassifier(**params)
# generate more samples
generateFDM(XGB)
