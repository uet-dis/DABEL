"""
Script sinh 4800 mẫu synthetic sử dụng TabGAN (CTGAN-based)
Cấu hình tối ưu nhất cho chất lượng dữ liệu sinh ra
Output: dataset/TabGAN_augmented_9600.csv (4800 original + 4800 synthetic)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
import psutil

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from tabgan.sampler import GANGenerator

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Paths
    'train_path': 'train_shap_65.csv',
    'test_path': 'test_shap_65.csv',
    'output_path': 'dataset/TabGAN_augmented_9600.csv',
    
    # Generation settings
    'n_synthetic_samples': 4800,  # Số mẫu synthetic cần sinh
    
    # TabGAN parameters - Tối ưu cho chất lượng
    'batch_size': 512,       # Batch size cho CTGAN (sẽ được điều chỉnh chia hết cho 10)
    'epochs': 500,           # Số epochs training
    'patience': 50,          # Early stopping patience
    
    # Adversarial model params (LightGBM)
    'adversarial_params': {
        'metrics': 'AUC',
        'max_depth': 15,
        'max_bin': 255,
        'learning_rate': 0.03,
        'random_state': 42,
        'n_estimators': 1000,
        'num_leaves': 100,
    },
    
    # Random seed
    'seed': 42,
}

# Set seeds
np.random.seed(CONFIG['seed'])


def setup_gpu():
    """Cấu hình GPU"""
    print("=" * 80)
    print("TABGAN - SYNTHETIC DATA GENERATION")
    print("=" * 80)
    
    # PyTorch GPU check (TabGAN uses CTGAN which is PyTorch-based)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\n[System Info]")
        print(f"  CPU Cores: {psutil.cpu_count(logical=True)}")
        print(f"  RAM Usage: {psutil.virtual_memory().percent}%")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  PyTorch CUDA: {cuda_available}")
        if cuda_available:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[WARN] PyTorch not available")
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


def prepare_data(train, test):
    """Chuẩn bị dữ liệu cho TabGAN"""
    print("\n[2/7] Preparing data for TabGAN...")
    
    X_train = train.drop(['Label'], axis=1)
    y_train = train['Label'].astype(int)
    
    X_test = test.drop(['Label'], axis=1)
    y_test = test['Label'].astype(int)
    
    feature_names = X_train.columns.tolist()
    
    # StandardScaler cho TabGAN
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names
    ).astype(np.float32)
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names
    ).astype(np.float32)
    
    print(f"  Features: {len(feature_names)}")
    print(f"  Train samples: {len(X_train_scaled)}")
    print(f"  Classes: {sorted(y_train.unique())}")
    
    return X_train_scaled, y_train.values, X_test_scaled, y_test.values, scaler, feature_names


def generate_synthetic_tabgan(X_train_scaled, y_train, X_test_scaled, n_samples, config):
    """Sinh dữ liệu synthetic với TabGAN"""
    print(f"\n[3/7] Generating {n_samples} synthetic samples with TabGAN...")
    
    # Tính gen_x_times để sinh đúng số lượng mẫu cần
    original_size = len(y_train)
    gen_x_times = n_samples / original_size
    
    # Điều chỉnh batch_size chia hết cho 10 (yêu cầu của CTGAN)
    batch_size = config['batch_size']
    pack = 10
    batch_size_adj = (batch_size // pack) * pack
    if batch_size_adj < pack:
        batch_size_adj = pack
    
    print(f"  gen_x_times: {gen_x_times:.2f}")
    print(f"  batch_size: {batch_size_adj}")
    print(f"  epochs: {config['epochs']}")
    print(f"  patience: {config['patience']}")
    
    # Prepare DataFrames cho TabGAN
    y_train_df = pd.DataFrame(y_train, columns=["Label"])
    X_train_df = X_train_scaled.copy()
    X_test_df = X_test_scaled.copy()
    
    start_time = time.time()
    
    # TabGAN Generator
    gan = GANGenerator(
        gen_x_times=gen_x_times,
        cat_cols=None,
        bot_filter_quantile=0.001,
        top_filter_quantile=0.999,
        is_post_process=True,
        adversarial_model_params=config['adversarial_params'],
        pregeneration_frac=2,
        only_generated_data=True,  # Chỉ lấy synthetic data
        gen_params={
            "batch_size": batch_size_adj,
            "patience": config['patience'],
            "epochs": config['epochs'],
        },
    )
    
    print(f"  Training CTGAN... (this may take a while)")
    
    X_syn, y_syn = gan.generate_data_pipe(
        X_train_df,
        y_train_df,
        X_test_df,
        deep_copy=True,
        only_adversarial=False,
        use_adversarial=True,
    )
    
    gen_time = time.time() - start_time
    print(f"  ✓ Generation completed in {gen_time:.2f}s ({gen_time/60:.2f}m)")
    
    # Convert to numpy
    X_syn_np = X_syn.values if hasattr(X_syn, 'values') else X_syn
    y_syn_np = y_syn.values.reshape(-1) if hasattr(y_syn, 'values') else y_syn.reshape(-1)
    
    # Nếu sinh nhiều hơn cần, cắt bớt
    if len(y_syn_np) > n_samples:
        indices = np.random.choice(len(y_syn_np), n_samples, replace=False)
        X_syn_np = X_syn_np[indices]
        y_syn_np = y_syn_np[indices]
    
    print(f"  Synthetic samples: {len(y_syn_np)}")
    
    # Label distribution
    print(f"  Synthetic label distribution:")
    unique, counts = np.unique(y_syn_np, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    Class {int(label)}: {count} ({count/len(y_syn_np)*100:.1f}%)")
    
    return X_syn_np, y_syn_np.astype(int)


def inverse_transform_synthetic(X_syn, scaler, feature_names):
    """Inverse transform synthetic data về scale gốc"""
    print("\n[4/7] Inverse transforming synthetic data...")
    
    X_syn_original = scaler.inverse_transform(X_syn)
    syn_df = pd.DataFrame(X_syn_original, columns=feature_names)
    
    print(f"  ✓ Inverse transform completed")
    
    return syn_df


def validate_quality(train_orig, synthetic_df, synthetic_y, test, feature_names):
    """Kiểm tra chất lượng synthetic data"""
    print("\n[5/7] Validating synthetic data quality...")
    
    orig_features = train_orig.drop(['Label'], axis=1)
    
    # Check duplicates
    print("\n  Checking for duplicates...")
    duplicates = 0
    sample_size = min(500, len(synthetic_df))
    for idx in range(sample_size):
        syn_row = synthetic_df.iloc[idx].values
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
    et_syn = ExtraTreesClassifier(
        n_estimators=130, max_leaf_nodes=15000, n_jobs=-1,
        random_state=0, bootstrap=True, criterion='entropy'
    )
    et_syn.fit(synthetic_df, synthetic_y)
    acc_syn = accuracy_score(y_test, et_syn.predict(X_test))
    
    print(f"    Accuracy (original only): {acc_orig:.4f}")
    print(f"    Accuracy (synthetic only): {acc_syn:.4f}")
    print(f"    Gap: {abs(acc_orig - acc_syn):.4f}")
    
    return dup_rate, acc_orig, acc_syn


def create_augmented_dataset(train_orig, synthetic_df, synthetic_y, output_path):
    """Tạo và lưu dataset augmented"""
    print(f"\n[6/7] Creating augmented dataset...")
    
    # Add label to synthetic
    synthetic_df = synthetic_df.copy()
    synthetic_df['Label'] = synthetic_y
    
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
    
    # Unique predictions
    unique_preds, counts = np.unique(pred, return_counts=True)
    print(f"  Predicted classes: {unique_preds.tolist()}")
    print(f"  Predictions per class: {counts.tolist()}")
    
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
    
    setup_gpu()
    
    # Load data
    train, test = load_data(CONFIG['train_path'], CONFIG['test_path'])
    
    # Prepare
    X_train_scaled, y_train, X_test_scaled, y_test, scaler, feature_names = prepare_data(train, test)
    
    # Generate synthetic
    X_syn, y_syn = generate_synthetic_tabgan(
        X_train_scaled, y_train, X_test_scaled,
        CONFIG['n_synthetic_samples'],
        CONFIG
    )
    
    # Inverse transform
    synthetic_df = inverse_transform_synthetic(X_syn, scaler, feature_names)
    
    # Validate
    dup_rate, acc_orig, acc_syn = validate_quality(
        train, synthetic_df, y_syn, test, feature_names
    )
    
    # Create augmented dataset
    augmented_df = create_augmented_dataset(
        train, synthetic_df, y_syn, CONFIG['output_path']
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
    print("✓ COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
