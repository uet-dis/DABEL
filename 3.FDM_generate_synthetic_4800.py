"""
Script sinh 4800 mẫu synthetic sử dụng Forest Diffusion Model (FDM)
Cấu hình tối ưu nhất cho chất lượng dữ liệu sinh ra
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from ForestDiffusion import ForestDiffusionModel as ForestFlowModel
import time
import psutil

# =============================================================================
# CONFIGURATION - Tối ưu cho chất lượng cao nhất
# =============================================================================
CONFIG = {
    # Paths
    'train_path': 'train_shap_65.csv',
    'test_path': 'test_shap_65.csv',
    'output_path': 'dataset/FDM_augmented_9600.csv',  # 4800 original + 4800 synthetic
    
    # Generation settings
    'n_synthetic_samples': 4800,  # Số mẫu synthetic cần sinh
    
    # FDM parameters - Tối ưu cho chất lượng
    'fdm_params': {
        'n_t': 150,           # Số bước diffusion (cao hơn = chất lượng tốt hơn)
        'duplicate_K': 300,   # Số lần duplicate (cao hơn = đa dạng hơn)
        'diffusion_type': 'flow',  # 'flow' cho kết quả tốt hơn 'vp'
        'n_jobs': -1,         # Sử dụng tất cả CPU cores
        'seed': 42,           # Reproducibility
    },
    
    # Validation model
    'use_gpu': True,
}

def get_system_info():
    """Hiển thị thông tin hệ thống"""
    print("=" * 80)
    print("FOREST DIFFUSION MODEL - SYNTHETIC DATA GENERATION")
    print("=" * 80)
    print(f"\n[System Info]")
    print(f"  CPU Cores: {psutil.cpu_count(logical=True)}")
    print(f"  RAM Usage: {psutil.virtual_memory().percent}%")
    print(f"  RAM Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', 
                               '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  GPU: {result.stdout.strip()}")
    except:
        print("  GPU: Not available")
    print()


def load_data(train_path, test_path):
    """Load và xử lý dữ liệu"""
    print("[1/6] Loading data...")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"  Original train: {train.shape[0]} samples, {train.shape[1]} features")
    print(f"  Test: {test.shape[0]} samples")
    
    # Phân bố nhãn
    label_dist = train['Label'].value_counts().sort_index()
    print(f"  Label distribution:")
    for label, count in label_dist.items():
        print(f"    Class {label}: {count} samples ({count/len(train)*100:.1f}%)")
    
    return train, test


def preprocess_data(train, test):
    """Tiền xử lý dữ liệu"""
    print("\n[2/6] Preprocessing data...")
    
    X_train = train.drop(['Label'], axis=1)
    y_train = train['Label'].astype(int)
    X_test = test.drop(['Label'], axis=1)
    y_test = test['Label'].astype(int)
    
    y_train_enc = y_train.values
    y_test_enc = y_test.values
    
    n_classes = len(np.unique(y_train_enc))
    print(f"  Labels already encoded as integers: {np.unique(y_train_enc)}")
    
    # MinMax scaling (phù hợp với FDM)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Float32 cho hiệu quả bộ nhớ
    X_train_np = X_train_scaled.astype(np.float32)
    X_test_np = X_test_scaled.astype(np.float32)
    
    print(f"  Scaled features to [0, 1] range")
    print(f"  Number of classes: {n_classes}")
    
    return X_train_np, y_train_enc, X_test_np, y_test_enc, n_classes, scaler, X_train.columns


def train_fdm_model(X_train, y_train, fdm_params):
    """Train Forest Diffusion Model"""
    print("\n[3/6] Training Forest Diffusion Model...")
    print(f"  Parameters:")
    for k, v in fdm_params.items():
        print(f"    {k}: {v}")
    
    start_time = time.time()
    
    fdm = ForestFlowModel(
        X=X_train,
        label_y=y_train,
        **fdm_params
    )
    
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f}s ({train_time/60:.2f}m)")
    
    return fdm


def generate_synthetic_data(fdm, n_samples, feature_names, n_classes):
    """Sinh dữ liệu synthetic với FDM"""
    print(f"\n[4/6] Generating {n_samples} synthetic samples...")
    
    start_time = time.time()
    
    # Sinh theo batch để đảm bảo đa dạng
    batch_size = min(2400, n_samples)  # Batch nhỏ hơn để đa dạng hơn
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_synthetic = []
    
    for i in range(n_batches):
        current_batch_size = min(batch_size, n_samples - len(all_synthetic) * batch_size if all_synthetic else batch_size)
        if i == n_batches - 1:
            current_batch_size = n_samples - sum(len(b) for b in all_synthetic)
        
        print(f"  Generating batch {i+1}/{n_batches} ({current_batch_size} samples)...")
        
        # Generate với seed khác nhau cho mỗi batch
        Xy_gen = fdm.generate(batch_size=current_batch_size)
        
        # Tách X và y
        X_gen = Xy_gen[:, :-1]
        y_gen = Xy_gen[:, -1]
        
        # Tạo DataFrame
        batch_df = pd.DataFrame(X_gen, columns=feature_names)
        batch_df['Label'] = y_gen.astype(int)
        
        all_synthetic.append(batch_df)
    
    synthetic_df = pd.concat(all_synthetic, ignore_index=True)
    
    gen_time = time.time() - start_time
    print(f"  Generation completed in {gen_time:.2f}s ({gen_time/60:.2f}m)")
    
    # Phân bố nhãn synthetic
    syn_dist = synthetic_df['Label'].value_counts().sort_index()
    print(f"  Synthetic label distribution:")
    for label, count in syn_dist.items():
        print(f"    Class {label}: {count} samples ({count/len(synthetic_df)*100:.1f}%)")
    
    return synthetic_df


def validate_quality(train_orig, synthetic_df, X_test, y_test):
    """Kiểm tra chất lượng dữ liệu synthetic"""
    print("\n[5/6] Validating synthetic data quality...")
    
    # 1. Kiểm tra duplicate với original
    print("\n  Checking for duplicates with original data...")
    
    orig_features = train_orig.drop(['Label'], axis=1)
    syn_features = synthetic_df.drop(['Label'], axis=1)
    
    duplicates = 0
    sample_size = min(500, len(synthetic_df))
    for idx in range(sample_size):
        syn_row = syn_features.iloc[idx].values
        # So sánh với tolerance nhỏ
        matches = np.all(np.isclose(orig_features.values, syn_row, rtol=1e-5), axis=1)
        if np.any(matches):
            duplicates += 1
    
    dup_rate = duplicates / sample_size * 100
    print(f"    Duplicate rate (checked {sample_size} samples): {dup_rate:.2f}%")
    
    if dup_rate > 10:
        print("    ⚠️  Warning: High duplicate rate! Consider increasing n_t or duplicate_K")
    else:
        print("    ✓ Duplicate rate is acceptable")
    
    # 2. So sánh thống kê
    print("\n  Comparing statistical properties...")
    
    orig_mean = orig_features.mean()
    syn_mean = syn_features.mean()
    mean_diff = np.abs(orig_mean - syn_mean).mean()
    
    orig_std = orig_features.std()
    syn_std = syn_features.std()
    std_diff = np.abs(orig_std - syn_std).mean()
    
    print(f"    Mean absolute difference (means): {mean_diff:.6f}")
    print(f"    Mean absolute difference (stds): {std_diff:.6f}")
    
    # 3. Test model performance
    print("\n  Testing model performance...")
    
    # Scaler cho test
    scaler = MinMaxScaler()
    X_train_orig = scaler.fit_transform(orig_features)
    X_test_scaled = scaler.transform(X_test)

    y_train_orig = train_orig['Label'].astype(int).values
    y_test_enc = y_test.astype(int).values
    
    # Model on original
    et_orig = ExtraTreesClassifier(
        n_estimators=130,
        max_leaf_nodes=15000,
        n_jobs=-1,
        random_state=0,
        bootstrap=True,
        criterion='entropy'
    )
    et_orig.fit(X_train_orig, y_train_orig)
    pred_orig = et_orig.predict(X_test_scaled)
    acc_orig = accuracy_score(y_test_enc, pred_orig)
    
    # Model on synthetic only
    X_syn = scaler.fit_transform(syn_features)
    y_syn = synthetic_df['Label'].astype(int).values
    
    et_syn = ExtraTreesClassifier(
        n_estimators=130,
        max_leaf_nodes=15000,
        n_jobs=-1,
        random_state=0,
        bootstrap=True,
        criterion='entropy'
    )
    et_syn.fit(X_syn, y_syn)
    
    X_test_scaled_syn = scaler.transform(X_test)
    pred_syn = et_syn.predict(X_test_scaled_syn)
    acc_syn = accuracy_score(y_test_enc, pred_syn)
    
    print(f"    Accuracy (original data only): {acc_orig:.4f}")
    print(f"    Accuracy (synthetic data only): {acc_syn:.4f}")
    print(f"    Accuracy gap: {abs(acc_orig - acc_syn):.4f}")
    
    if abs(acc_orig - acc_syn) < 0.05:
        print("    ✓ Synthetic data quality is excellent!")
    elif abs(acc_orig - acc_syn) < 0.10:
        print("    ✓ Synthetic data quality is good")
    else:
        print("    ⚠️  Warning: Large accuracy gap, consider tuning FDM parameters")
    
    return dup_rate, mean_diff, acc_orig, acc_syn


def create_augmented_dataset(train_orig, synthetic_df, output_path, scaler, feature_names):
    """Tạo dataset augmented và lưu file"""
    print(f"\n[6/6] Creating augmented dataset...")
    
    # Inverse transform synthetic data để có giá trị gốc
    syn_features = synthetic_df.drop(['Label'], axis=1)
    syn_labels = synthetic_df['Label']
    
    # Inverse scale
    syn_features_original = scaler.inverse_transform(syn_features)
    syn_df_original = pd.DataFrame(syn_features_original, columns=feature_names)
    syn_df_original['Label'] = syn_labels.values
    
    # Gộp với dữ liệu gốc
    augmented_df = pd.concat([train_orig, syn_df_original], ignore_index=True)
    
    print(f"  Original samples: {len(train_orig)}")
    print(f"  Synthetic samples: {len(syn_df_original)}")
    print(f"  Total augmented samples: {len(augmented_df)}")
    
    # Shuffle để trộn đều
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Phân bố nhãn cuối cùng
    final_dist = augmented_df['Label'].value_counts().sort_index()
    print(f"\n  Final label distribution:")
    for label, count in final_dist.items():
        print(f"    Class {label}: {count} samples ({count/len(augmented_df)*100:.1f}%)")
    
    # Lưu file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved augmented dataset to: {output_path}")
    
    return augmented_df


def test_augmented_model(augmented_df, test_df, train_orig):
    """Test model trên dữ liệu augmented"""
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    X_aug = augmented_df.drop(['Label'], axis=1)
    y_aug = augmented_df['Label'].astype(int)
    
    X_test = test_df.drop(['Label'], axis=1)
    y_test = test_df['Label'].astype(int)
    
    # Scale
    scaler = MinMaxScaler()
    X_aug_scaled = scaler.fit_transform(X_aug)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ExtraTrees
    print("\n[ExtraTrees Classifier]")
    et = ExtraTreesClassifier(
        n_estimators=130,
        max_leaf_nodes=15000,
        n_jobs=-1,
        random_state=0,
        bootstrap=True,
        criterion='entropy'
    )
    et.fit(X_aug_scaled, y_aug)
    pred_et = et.predict(X_test_scaled)
    
    acc_et = accuracy_score(y_test, pred_et)
    prec_et = precision_score(y_test, pred_et, average='macro')
    rec_et = recall_score(y_test, pred_et, average='macro')
    f1_et = f1_score(y_test, pred_et, average='macro')
    
    print(f"  Accuracy:  {acc_et:.4f}")
    print(f"  Precision: {prec_et:.4f}")
    print(f"  Recall:    {rec_et:.4f}")
    print(f"  F1-Score:  {f1_et:.4f}")
    
    # Unique predictions
    unique_preds = np.unique(pred_et, return_counts=True)
    print(f"  Predicted classes: {unique_preds[0].tolist()}")
    print(f"  Predictions per class: {unique_preds[1].tolist()}")
    
    # So sánh với original
    print("\n[Comparison with Original Data Only]")
    X_orig = train_orig.drop(['Label'], axis=1)
    y_orig = train_orig['Label'].astype(int)
    
    scaler_orig = MinMaxScaler()
    X_orig_scaled = scaler_orig.fit_transform(X_orig)
    X_test_scaled_orig = scaler_orig.transform(X_test)
    
    et_orig = ExtraTreesClassifier(
        n_estimators=130,
        max_leaf_nodes=15000,
        n_jobs=-1,
        random_state=0,
        bootstrap=True,
        criterion='entropy'
    )
    et_orig.fit(X_orig_scaled, y_orig)
    pred_orig = et_orig.predict(X_test_scaled_orig)
    
    acc_orig = accuracy_score(y_test, pred_orig)
    print(f"  Original data accuracy: {acc_orig:.4f}")
    print(f"  Augmented data accuracy: {acc_et:.4f}")
    print(f"  Improvement: {(acc_et - acc_orig)*100:+.2f}%")
    
    return acc_et, acc_orig


def main():
    """Main function"""
    total_start = time.time()
    
    get_system_info()
    
    # Load data
    train, test = load_data(CONFIG['train_path'], CONFIG['test_path'])
    
    # Preprocess
    X_train, y_train, X_test, y_test, n_classes, scaler, feature_names = preprocess_data(train, test)
    
    # Train FDM
    fdm = train_fdm_model(X_train, y_train, CONFIG['fdm_params'])
    
    # Generate synthetic data
    synthetic_df = generate_synthetic_data(
        fdm, 
        CONFIG['n_synthetic_samples'], 
        feature_names,
        n_classes
    )
    
    # Validate quality
    dup_rate, mean_diff, acc_orig, acc_syn = validate_quality(
        train, 
        synthetic_df, 
        test.drop(['Label'], axis=1),
        test['Label']
    )
    
    # Create augmented dataset
    augmented_df = create_augmented_dataset(
        train, 
        synthetic_df, 
        CONFIG['output_path'],
        scaler,
        feature_names
    )
    
    # Test final model
    acc_aug, acc_baseline = test_augmented_model(augmented_df, test, train)
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total execution time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"  Original samples: {len(train)}")
    print(f"  Synthetic samples generated: {CONFIG['n_synthetic_samples']}")
    print(f"  Final augmented samples: {len(augmented_df)}")
    print(f"  Output file: {CONFIG['output_path']}")
    print(f"  Duplicate rate: {dup_rate:.2f}%")
    print(f"  Baseline accuracy: {acc_baseline:.4f}")
    print(f"  Augmented accuracy: {acc_aug:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
