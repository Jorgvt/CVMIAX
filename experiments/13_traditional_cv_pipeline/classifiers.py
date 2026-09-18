"""
Classifiers Module for Traditional Computer Vision Pipeline.

Wraps classical statistical and machine learning classifiers:
1. Support Vector Classifier (Linear & RBF Kernels)
2. Random Forest Classifier
3. k-Nearest Neighbors (k-NN)
"""

from typing import Dict, Any, Tuple, Optional
import time
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def get_classifier(classifier_type: str = 'svm_rbf', **kwargs) -> Any:
    """
    Factory function to instantiate traditional machine learning classifiers.
    
    Parameters
    ----------
    classifier_type : str
        One of 'svm_linear', 'svm_rbf', 'random_forest', 'knn'.
    **kwargs : dict
        Hyperparameters passed to the classifier constructor.
        
    Returns
    -------
    sklearn classifier instance.
    """
    if classifier_type == 'svm_linear':
        c_val = kwargs.get('C', 1.0)
        max_iter = kwargs.get('max_iter', 1000)
        tol = kwargs.get('tol', 1e-3)
        return LinearSVC(C=c_val, max_iter=max_iter, tol=tol, random_state=42, dual=False)
    
    elif classifier_type == 'svm_rbf':
        c_val = kwargs.get('C', 5.0)
        gamma = kwargs.get('gamma', 'scale')
        return SVC(C=c_val, kernel='rbf', gamma=gamma, random_state=42)
    
    elif classifier_type == 'random_forest':
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    
    elif classifier_type == 'knn':
        n_neighbors = kwargs.get('n_neighbors', 5)
        return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def train_and_evaluate_classifier(
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train a classifier and record performance, training time, and inference speed.
    
    Parameters
    ----------
    clf : sklearn classifier
    X_train : np.ndarray (N, D)
    y_train : np.ndarray (N,)
    X_test : np.ndarray (M, D)
    y_test : np.ndarray (M,)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'train_time_sec': float
        - 'test_time_sec': float
        - 'accuracy': float
        - 'y_pred': np.ndarray
        - 'confusion_matrix': np.ndarray
        - 'classification_report': str
    """
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    
    t1 = time.time()
    y_pred = clf.predict(X_test)
    test_time = time.time() - t1
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    
    return {
        'model': clf,
        'train_time_sec': train_time,
        'test_time_sec': test_time,
        'accuracy': float(acc),
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'classification_report': report
    }
