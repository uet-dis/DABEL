"""
Script sinh 4800 mẫu synthetic sử dụng Conditional Flow Matching (CFM)
Cấu hình tối ưu nhất cho chất lượng dữ liệu sinh ra
Output: dataset/CFM_augmented_9600.csv (4800 original + 4800 synthetic)
"""

import copy
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from concurrent.futures import ThreadPoolExecutor
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import time
import psutil

# =============================================================================
# AUTO-DETECT GPU
# =============================================================================
def get_xgb_device():
    """Auto-detect GPU availability for XGBoost"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode == 0 and torch.cuda.is_available():
            return "cuda"
    except:
        pass
    return "cpu"

XGB_DEVICE = get_xgb_device()
print(f"XGBoost device: {XGB_DEVICE}")

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Paths
    'train_path': 'train_shap_65.csv',
    'test_path': 'test_shap_65.csv',
    'output_path': 'dataset/CFM_augmented_9600.csv',
    
    # Generation settings
    'n_synthetic_samples': 4800,  # Số mẫu synthetic cần sinh
    
    # CFM parameters - Tối ưu cho chất lượng
    'n_t': 100,           # Số bước flow (cao hơn = chất lượng tốt hơn)
    'duplicate_K': 300,   # Số lần duplicate (cao hơn = đa dạng hơn)
    
    # XGBoost parameters
    'max_depth': 10,
    'n_estimators': 200,
    'eta': 0.1,
    'reg_lambda': 1.0,
    'reg_alpha': 0.5,
    'subsample': 0.8,
    
    # Parallel workers
    'n_workers': 32,
    
    # Random seed
    'seed': 1980,
}

# Set seeds for reproducibility
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
torch.cuda.manual_seed(CONFIG['seed'])
torch.cuda.manual_seed_all(CONFIG['seed'])
torch.backends.cudnn.benchmark = True


def get_system_info():
    """Hiển thị thông tin hệ thống"""
    print("=" * 80)
    print("CONDITIONAL FLOW MATCHING - SYNTHETIC DATA GENERATION")
    print("=" * 80)
    print(f"\n[System Info]")
    print(f"  CPU Cores: {psutil.cpu_count(logical=True)}")
    print(f"  RAM Usage: {psutil.virtual_memory().percent}%")
    print(f"  PyTorch CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()


def load_data(train_path, test_path):
    """Load dữ liệu"""
    print("[1/7] Loading data...")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"  Original train: {train.shape[0]} samples, {train.shape[1]} features")
    print(f"  Test: {test.shape[0]} samples")
    
    # Phân bố nhãn
    label_dist = train['Label'].value_counts().sort_index()
    print(f"  Label distribution:")
    for label, count in label_dist.items():
        print(f"    Class {int(label)}: {count} samples ({count/len(train)*100:.1f}%)")
    
    return train, test


def prepare_data(train):
    """Chuẩn bị dữ liệu cho CFM"""
    print("\n[2/7] Preparing data for CFM...")
    
    X = train.drop(['Label'], axis=1)
    y = train['Label'].astype(int)
    
    feature_names = X.columns.tolist()
    X = X.to_numpy()
    y = np.array(y)
    
    # Shuffle
    perm = np.random.permutation(X.shape[0])
    np.take(X, perm, axis=0, out=X)
    np.take(y, perm, axis=0, out=y)
    
    # Save original data
    X_true, y_true = copy.deepcopy(X), copy.deepcopy(y)
    
    # Min/Max values
    X_min = np.nanmin(X, axis=0, keepdims=1)
    X_max = np.nanmax(X, axis=0, keepdims=1)
    
    # Min-Max scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Classes: {np.unique(y)}")
    
    return X_scaled, y, X_true, y_true, X_min, X_max, scaler, feature_names


def build_flow_data(X_scaled, y, n_t, duplicate_K):
    """Xây dựng dữ liệu flow matching"""
    print(f"\n[3/7] Building flow matching data...")
    print(f"  n_t={n_t}, duplicate_K={duplicate_K}")
    
    b, c = X_scaled.shape
    
    # Duplicate data
    X1 = np.tile(X_scaled, (duplicate_K, 1))
    X0 = np.random.normal(size=X1.shape)
    
    print(f"  Total training samples: {X0.shape[0]} (original: {b})")
    
    # Class masks
    y_uniques, y_probs = np.unique(y, return_counts=True)
    y_probs = y_probs / np.sum(y_probs)
    
    mask_y = {}
    for i, label in enumerate(y_uniques):
        mask_y[label] = np.zeros(b, dtype=bool)
        mask_y[label][y == label] = True
        mask_y[label] = np.tile(mask_y[label], duplicate_K)
    
    # CFM
    FM = ConditionalFlowMatcher(sigma=0.0)
    t_levels = np.linspace(1e-3, 1, num=n_t)
    
    # Build X_train and y_train
    X_train = np.zeros((n_t, X0.shape[0], X0.shape[1]))
    y_train = np.zeros((n_t, X0.shape[0], X0.shape[1]))
    
    for i in range(n_t):
        t = torch.ones(X0.shape[0]) * t_levels[i]
        _, xt, ut = FM.sample_location_and_conditional_flow(
            torch.from_numpy(X0), torch.from_numpy(X1), t=t
        )
        X_train[i], y_train[i] = xt.numpy(), ut.numpy()
    
    return X_train, y_train, y_uniques, y_probs, mask_y, b, c


def train_xgb_models(X_train, y_train, y_uniques, mask_y, b, c, n_t, config):
    """Train XGBoost models"""
    print(f"\n[4/7] Training XGBoost models...")
    
    n_estimators = config['n_estimators']
    eta = config['eta']
    max_depth = config['max_depth']
    reg_lambda = config['reg_lambda']
    reg_alpha = config['reg_alpha']
    subsample = config['subsample']
    duplicate_K = config['duplicate_K']
    n_workers = config['n_workers']
    
    def train_parallel(args):
        i, j, k = args
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            objective="reg:squarederror",
            eta=eta,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            seed=666,
            device=XGB_DEVICE,
        )
        model.fit(
            X_train.reshape(n_t, b * duplicate_K, c)[i][mask_y[j], :],
            y_train.reshape(n_t, b * duplicate_K, c)[i][mask_y[j], k]
        )
        return model
    
    args_list = [(i, j, k) for i in range(n_t) for j in y_uniques for k in range(c)]
    
    print(f"  Total models: {len(args_list)} (n_t={n_t} × classes={len(y_uniques)} × features={c})")
    print(f"  Using {n_workers} parallel workers with GPU")
    print(f"  Training... (this may take a while)")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        models = list(executor.map(train_parallel, args_list))
    
    train_time = time.time() - start_time
    print(f"  ✓ Training completed in {train_time:.2f}s ({train_time/60:.2f}m)")
    
    # Reorganize models
    regr = [[[None for k in range(c)] for i in range(n_t)] for j in y_uniques]
    current_i = 0
    for i in range(n_t):
        for j_idx, j in enumerate(y_uniques):
            for k in range(c):
                regr[j_idx][i][k] = models[current_i]
                current_i += 1
    
    return regr


def generate_synthetic(regr, y_uniques, y_probs, scaler, X_min, X_max, n_samples, n_t, c):
    """Sinh dữ liệu synthetic"""
    print(f"\n[5/7] Generating {n_samples} synthetic samples...")
    
    def my_model(t, xt, mask_y=None):
        xt = xt.reshape(xt.shape[0] // c, c)
        out = np.zeros(xt.shape)
        i = int(round(t * (n_t - 1)))
        for j_idx, label in enumerate(y_uniques):
            for k in range(c):
                if np.any(mask_y[label]):
                    out[mask_y[label], k] = regr[j_idx][i][k].predict(xt[mask_y[label], :])
        return out.reshape(-1)
    
    def euler_solve(x0, my_model, N=100):
        h = 1 / (N - 1)
        x_fake = x0
        t = 0
        for i in range(N - 1):
            x_fake = x_fake + h * my_model(t=t, xt=x_fake)
            t = t + h
        return x_fake
    
    start_time = time.time()
    
    # Generate noise
    x0 = np.random.normal(size=(n_samples, c))
    
    # Generate random labels
    label_y_fake = y_uniques[np.argmax(np.random.multinomial(1, y_probs, size=n_samples), axis=1)]
    
    mask_y_fake = {}
    for label in y_uniques:
        mask_y_fake[label] = np.zeros(n_samples, dtype=bool)
        mask_y_fake[label][label_y_fake == label] = True
    
    print(f"  Running ODE solver with {n_t} steps...")
    
    # ODE solve
    ode_solved = euler_solve(
        my_model=partial(my_model, mask_y=mask_y_fake),
        x0=x0.reshape(-1),
        N=n_t
    )
    solution = ode_solved.reshape(n_samples, c)
    
    # Inverse transform
    solution = scaler.inverse_transform(solution)
    
    # Clip to min/max
    small = (solution < X_min).astype(float)
    solution = small * X_min + (1 - small) * solution
    big = (solution > X_max).astype(float)
    solution = big * X_max + (1 - big) * solution
    
    gen_time = time.time() - start_time
    print(f"  ✓ Generation completed in {gen_time:.2f}s ({gen_time/60:.2f}m)")
    
    # Label distribution
    print(f"  Synthetic label distribution:")
    for label in y_uniques:
        count = np.sum(label_y_fake == label)
        print(f"    Class {int(label)}: {count} samples ({count/n_samples*100:.1f}%)")
    
    return solution, label_y_fake


def validate_quality(train_orig, synthetic_X, synthetic_y, test, feature_names):
    """Kiểm tra chất lượng synthetic data"""
    print("\n[6/7] Validating synthetic data quality...")
    
    orig_features = train_orig.drop(['Label'], axis=1)
    
    # Check duplicates
    print("\n  Checking for duplicates...")
    duplicates = 0
    sample_size = min(500, len(synthetic_X))
    for idx in range(sample_size):
        syn_row = synthetic_X[idx]
        matches = np.all(np.isclose(orig_features.values, syn_row, rtol=1e-5), axis=1)
        if np.any(matches):
            duplicates += 1
    
    dup_rate = duplicates / sample_size * 100
    print(f"    Duplicate rate: {dup_rate:.2f}%")
    
    if dup_rate > 10:
        print("    ⚠️  Warning: High duplicate rate!")
    else:
        print("    ✓ Duplicate rate is acceptable")
    
    # Model performance
    print("\n  Testing model performance...")
    
    X_test = test.drop(['Label'], axis=1)
    y_test = test['Label'].astype(int)
    
    # Original only
    X_orig = train_orig.drop(['Label'], axis=1)
    y_orig = train_orig['Label'].astype(int)
    
    et_orig = ExtraTreesClassifier(
        n_estimators=130, max_leaf_nodes=15000, n_jobs=-1,
        random_state=0, bootstrap=True, criterion='entropy'
    )
    et_orig.fit(X_orig, y_orig)
    acc_orig = accuracy_score(y_test, et_orig.predict(X_test))
    
    # Synthetic only
    syn_df = pd.DataFrame(synthetic_X, columns=feature_names)
    et_syn = ExtraTreesClassifier(
        n_estimators=130, max_leaf_nodes=15000, n_jobs=-1,
        random_state=0, bootstrap=True, criterion='entropy'
    )
    et_syn.fit(syn_df, synthetic_y.astype(int))
    acc_syn = accuracy_score(y_test, et_syn.predict(X_test))
    
    print(f"    Accuracy (original only): {acc_orig:.4f}")
    print(f"    Accuracy (synthetic only): {acc_syn:.4f}")
    print(f"    Gap: {abs(acc_orig - acc_syn):.4f}")
    
    return dup_rate, acc_orig, acc_syn


def create_augmented_dataset(train_orig, synthetic_X, synthetic_y, feature_names, output_path):
    """Tạo và lưu dataset augmented"""
    print(f"\n[7/7] Creating augmented dataset...")
    
    # Create synthetic DataFrame
    synthetic_df = pd.DataFrame(synthetic_X, columns=feature_names)
    synthetic_df['Label'] = synthetic_y.astype(int)
    
    # Merge
    augmented_df = pd.concat([train_orig, synthetic_df], ignore_index=True)
    
    # Shuffle
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Original samples: {len(train_orig)}")
    print(f"  Synthetic samples: {len(synthetic_df)}")
    print(f"  Total augmented: {len(augmented_df)}")
    
    # Label distribution
    print(f"\n  Final label distribution:")
    for label in sorted(augmented_df['Label'].unique()):
        count = (augmented_df['Label'] == label).sum()
        print(f"    Class {int(label)}: {count} ({count/len(augmented_df)*100:.1f}%)")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved to: {output_path}")
    
    return augmented_df


def final_evaluation(augmented_df, test, train_orig):
    """Đánh giá cuối cùng"""
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    X_aug = augmented_df.drop(['Label'], axis=1)
    y_aug = augmented_df['Label'].astype(int)
    
    X_test = test.drop(['Label'], axis=1)
    y_test = test['Label'].astype(int)
    
    # Train on augmented
    et = ExtraTreesClassifier(
        n_estimators=130, max_leaf_nodes=15000, n_jobs=-1,
        random_state=0, bootstrap=True, criterion='entropy'
    )
    et.fit(X_aug, y_aug)
    pred = et.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average='macro')
    rec = recall_score(y_test, pred, average='macro')
    f1 = f1_score(y_test, pred, average='macro')
    
    print(f"\n[ExtraTrees on Augmented Data]")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Baseline
    X_orig = train_orig.drop(['Label'], axis=1)
    y_orig = train_orig['Label'].astype(int)
    
    et_orig = ExtraTreesClassifier(
        n_estimators=130, max_leaf_nodes=15000, n_jobs=-1,
        random_state=0, bootstrap=True, criterion='entropy'
    )
    et_orig.fit(X_orig, y_orig)
    acc_orig = accuracy_score(y_test, et_orig.predict(X_test))
    
    print(f"\n[Comparison]")
    print(f"  Original accuracy:  {acc_orig:.4f}")
    print(f"  Augmented accuracy: {acc:.4f}")
    print(f"  Improvement: {(acc - acc_orig)*100:+.2f}%")
    
    return acc, acc_orig


def main():
    total_start = time.time()
    
    get_system_info()
    
    # Load data
    train, test = load_data(CONFIG['train_path'], CONFIG['test_path'])
    
    # Prepare
    X_scaled, y, X_true, y_true, X_min, X_max, scaler, feature_names = prepare_data(train)
    
    # Build flow data
    X_train, y_train, y_uniques, y_probs, mask_y, b, c = build_flow_data(
        X_scaled, y, CONFIG['n_t'], CONFIG['duplicate_K']
    )
    
    # Train models
    regr = train_xgb_models(
        X_train, y_train, y_uniques, mask_y, b, c, CONFIG['n_t'], CONFIG
    )
    
    # Generate synthetic
    synthetic_X, synthetic_y = generate_synthetic(
        regr, y_uniques, y_probs, scaler, X_min, X_max,
        CONFIG['n_synthetic_samples'], CONFIG['n_t'], c
    )
    
    # Validate
    dup_rate, acc_orig, acc_syn = validate_quality(
        train, synthetic_X, synthetic_y, test, feature_names
    )
    
    # Create augmented dataset
    augmented_df = create_augmented_dataset(
        train, synthetic_X, synthetic_y, feature_names, CONFIG['output_path']
    )
    
    # Final evaluation
    acc_aug, acc_baseline = final_evaluation(augmented_df, test, train)
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"  Original samples: {len(train)}")
    print(f"  Synthetic samples: {CONFIG['n_synthetic_samples']}")
    print(f"  Total augmented: {len(augmented_df)}")
    print(f"  Output: {CONFIG['output_path']}")
    print(f"  Duplicate rate: {dup_rate:.2f}%")
    print(f"  Baseline accuracy: {acc_baseline:.4f}")
    print(f"  Augmented accuracy: {acc_aug:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
