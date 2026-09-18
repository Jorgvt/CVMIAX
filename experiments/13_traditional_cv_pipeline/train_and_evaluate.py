"""
Training & Evaluation Pipeline for Traditional Computer Vision.

Executes comparative benchmarks across multiple feature representations 
(Raw Pixels, HOG, LBP, PCA-reduced HOG) and classical classifiers 
(k-NN, Random Forest, Linear SVM, RBF SVM).
"""

from typing import Dict, Any, Tuple
import os
import time
import numpy as np
from tensorflow.keras.datasets import mnist

from preprocessing import preprocess_batch
from features import extract_features_dataset
from feature_processing import FeatureProcessor, compute_scree_analysis
from classifiers import get_classifier, train_and_evaluate_classifier


def load_and_preprocess_dataset(
    n_train: int = 10000,
    n_test: int = 2000,
    deskew: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset and apply classical moments deskewing and normalization.
    
    Parameters
    ----------
    n_train : int
        Number of training samples to use for fast, rigorous benchmarking.
    n_test : int
        Number of test samples.
    deskew : bool
        Whether to apply moments-based deskewing.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (x_train_raw, x_train_proc, y_train, x_test_raw, x_test_proc, y_test)
    """
    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    
    # Subsample for pedagogical benchmarking
    x_train_raw = x_train_full[:n_train]
    y_train = y_train_full[:n_train]
    x_test_raw = x_test_full[:n_test]
    y_test = y_test_full[:n_test]
    
    print(f"--> Preprocessing {n_train} training images (Deskew={deskew})...")
    t0 = time.time()
    x_train_proc = preprocess_batch(x_train_raw, deskew=deskew, blur=True, sigma=0.5)
    t_pre_tr = time.time() - t0
    print(f"    Done in {t_pre_tr:.2f}s.")
    
    print(f"--> Preprocessing {n_test} test images...")
    t0 = time.time()
    x_test_proc = preprocess_batch(x_test_raw, deskew=deskew, blur=True, sigma=0.5)
    t_pre_te = time.time() - t0
    print(f"    Done in {t_pre_te:.2f}s.")
    
    return x_train_raw, x_train_proc, y_train, x_test_raw, x_test_proc, y_test


def run_benchmark_suite(
    n_train: int = 10000,
    n_test: int = 2000
) -> Dict[str, Any]:
    """
    Run full benchmark comparison across feature extractors and classifiers.
    """
    # 1. Load and preprocess
    x_tr_raw, x_tr_proc, y_tr, x_te_raw, x_te_proc, y_te = load_and_preprocess_dataset(
        n_train=n_train, n_test=n_test, deskew=True
    )
    
    # 2. Extract Feature Representations
    print("\n[Stage 2] Extracting Feature Representations:")
    
    # (a) Raw Pixels
    print("  -> Extracting Raw Pixels (784D)...")
    X_tr_raw = extract_features_dataset(x_tr_raw, feature_type='raw')
    X_te_raw = extract_features_dataset(x_te_raw, feature_type='raw')
    
    # (b) HOG Features
    print("  -> Extracting HOG Descriptors (144D)...")
    t0 = time.time()
    X_tr_hog = extract_features_dataset(x_tr_proc, feature_type='hog', pixels_per_cell=(7, 7), cells_per_block=(2, 2))
    X_te_hog = extract_features_dataset(x_te_proc, feature_type='hog', pixels_per_cell=(7, 7), cells_per_block=(2, 2))
    print(f"     Extracted in {time.time()-t0:.2f}s. Feature shape: {X_tr_hog.shape}")
    
    # (c) LBP Features
    print("  -> Extracting LBP Texture Histograms (10D)...")
    X_tr_lbp = extract_features_dataset(x_tr_proc, feature_type='lbp', num_points=8, radius=1, method='uniform')
    X_te_lbp = extract_features_dataset(x_te_proc, feature_type='lbp', num_points=8, radius=1, method='uniform')
    
    # 3. Feature Processing (Standardization & PCA)
    print("\n[Stage 3] Feature Processing & Dimensionality Reduction:")
    # Scree analysis
    _, cum_var_raw = compute_scree_analysis(X_tr_raw, max_components=50)
    _, cum_var_hog = compute_scree_analysis(X_tr_hog, max_components=50)
    
    # Standardize HOG
    proc_hog = FeatureProcessor(standardize=True, n_components=None)
    X_tr_hog_scaled = proc_hog.fit_transform(X_tr_hog)
    X_te_hog_scaled = proc_hog.transform(X_te_hog)
    
    # PCA on HOG (50 components)
    proc_hog_pca = FeatureProcessor(standardize=True, n_components=50)
    X_tr_hog_pca = proc_hog_pca.fit_transform(X_tr_hog)
    X_te_hog_pca = proc_hog_pca.transform(X_te_hog)
    print(f"  -> HOG + PCA (50 components) explained variance: {proc_hog_pca.cumulative_explained_variance[-1]*100:.2f}%")

    # Standardize Raw & LBP
    proc_raw = FeatureProcessor(standardize=True, n_components=None)
    X_tr_raw_scaled = proc_raw.fit_transform(X_tr_raw)
    X_te_raw_scaled = proc_raw.transform(X_te_raw)

    proc_lbp = FeatureProcessor(standardize=True, n_components=None)
    X_tr_lbp_scaled = proc_lbp.fit_transform(X_tr_lbp)
    X_te_lbp_scaled = proc_lbp.transform(X_te_lbp)
    
    # 4. Train & Evaluate Classifiers
    print("\n[Stage 4] Training & Evaluating Classifiers:")
    
    pipelines = [
        ("Raw Pixels + k-NN (k=5)", get_classifier('knn', n_neighbors=5), X_tr_raw_scaled, X_te_raw_scaled),
        ("Raw Pixels + Linear SVM", get_classifier('svm_linear', C=0.1), X_tr_raw_scaled, X_te_raw_scaled),
        ("LBP (10D) + SVM (RBF)", get_classifier('svm_rbf', C=5.0), X_tr_lbp_scaled, X_te_lbp_scaled),
        ("HOG + Random Forest", get_classifier('random_forest', n_estimators=100), X_tr_hog_scaled, X_te_hog_scaled),
        ("HOG + Linear SVM", get_classifier('svm_linear', C=1.0), X_tr_hog_scaled, X_te_hog_scaled),
        ("HOG + PCA (50D) + SVM (RBF)", get_classifier('svm_rbf', C=5.0), X_tr_hog_pca, X_te_hog_pca),
        ("HOG + SVM (RBF) [Gold Standard]", get_classifier('svm_rbf', C=5.0), X_tr_hog_scaled, X_te_hog_scaled),
    ]
    
    results = {}
    for name, clf, X_tr, X_te in pipelines:
        print(f"  -> Training {name}...")
        res = train_and_evaluate_classifier(clf, X_tr, y_tr, X_te, y_te)
        results[name] = res
        print(f"     Accuracy: {res['accuracy']*100:.2f}% | Train Time: {res['train_time_sec']:.2f}s | Test Time: {res['test_time_sec']:.2f}s")
        
    return {
        'results': results,
        'X_tr_raw': X_tr_raw,
        'X_tr_hog': X_tr_hog,
        'y_tr': y_tr,
        'cum_var_raw': cum_var_raw,
        'cum_var_hog': cum_var_hog,
        'x_te_raw': x_te_raw,
        'y_te': y_te,
        'best_predictions': results["HOG + SVM (RBF) [Gold Standard]"]['y_pred']
    }
