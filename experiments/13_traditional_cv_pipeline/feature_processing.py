"""
Feature Processing & Dimensionality Reduction Module.

This module provides mathematical feature transformation techniques:
1. Feature Standardization (Z-score scaling)
2. Principal Component Analysis (PCA) Eigen-decomposition & Projection
3. Scree Analysis (Explained variance spectrum)
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeatureProcessor:
    """
    Modular feature processor handling standardization, PCA, and transformation.
    """
    def __init__(self, standardize: bool = True, n_components: Optional[int] = None):
        """
        Parameters
        ----------
        standardize : bool
            Whether to apply zero-mean, unit-variance standardization.
        n_components : Optional[int]
            Number of principal components to keep (or None to retain all).
        """
        self.standardize = standardize
        self.n_components = n_components
        self.scaler = StandardScaler() if standardize else None
        self.pca = PCA(n_components=n_components) if n_components is not None else None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'FeatureProcessor':
        """
        Fit the standardization parameters and PCA projection matrix on training data.
        
        Parameters
        ----------
        X : np.ndarray
            Matrix of feature vectors of shape (N, D).
            
        Returns
        -------
        self
        """
        X_curr = X.copy()
        if self.standardize and self.scaler is not None:
            X_curr = self.scaler.fit_transform(X_curr)
        if self.pca is not None:
            self.pca.fit(X_curr)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new feature vectors using the fitted parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (M, D).
            
        Returns
        -------
        np.ndarray
            Transformed feature matrix of shape (M, k).
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureProcessor must be fitted before calling transform().")
        X_curr = X.copy()
        if self.standardize and self.scaler is not None:
            X_curr = self.scaler.transform(X_curr)
        if self.pca is not None:
            X_curr = self.pca.transform(X_curr)
        return X_curr.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        return self.fit(X).transform(X)

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Return the percentage of variance explained by each of the selected components.
        """
        if self.pca is not None and hasattr(self.pca, 'explained_variance_ratio_'):
            return self.pca.explained_variance_ratio_
        return None

    @property
    def cumulative_explained_variance(self) -> Optional[np.ndarray]:
        """
        Return the cumulative explained variance array.
        """
        if self.explained_variance_ratio is not None:
            return np.cumsum(self.explained_variance_ratio)
        return None


def compute_scree_analysis(X: np.ndarray, max_components: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform full PCA scree analysis to determine optimal dimensionality k.
    
    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix (N, D).
    max_components : int
        Number of top components to evaluate.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (individual_explained_variance, cumulative_explained_variance)
    """
    n_comp = min(max_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    
    indiv_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(indiv_var)
    return indiv_var, cum_var
