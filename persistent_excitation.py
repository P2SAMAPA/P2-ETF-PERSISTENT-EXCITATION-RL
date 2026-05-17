import numpy as np
from collections import deque

class PersistentExcitation:
    """
    Maintains a rolling covariance matrix of feature vectors and adjusts exploration noise.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self, feature_dim, lambda_min=0.01, window=1000):
        self.feature_dim = feature_dim
        self.lambda_min = lambda_min
        self.window = window
        self.buffer = deque(maxlen=window)
        # For incremental mean and covariance using Welford
        self.n = 0
        self.mean = np.zeros(feature_dim)
        self.M2 = np.zeros((feature_dim, feature_dim))  # sum of outer deviations
    
    def update(self, feature_vector):
        """Add new feature vector using Welford's algorithm."""
        self.buffer.append(feature_vector)
        self.n = min(self.n + 1, self.window)
        if self.n == 1:
            self.mean = feature_vector.copy()
            self.M2 = np.zeros((self.feature_dim, self.feature_dim))
        else:
            delta = feature_vector - self.mean
            self.mean += delta / self.n
            delta2 = feature_vector - self.mean
            self.M2 += np.outer(delta, delta2)
    
    def get_min_eigenvalue(self):
        """Return minimum eigenvalue of sample covariance (regularized)."""
        if self.n < self.feature_dim + 1:
            return 0.0
        # Compute sample covariance: M2 / (n - 1)
        cov = self.M2 / (self.n - 1)
        # Add small regularization to avoid singular/ill-conditioned matrix
        reg = 1e-6 * np.trace(cov) / self.feature_dim
        cov_reg = cov + reg * np.eye(self.feature_dim)
        try:
            eigvals = np.linalg.eigvalsh(cov_reg)
            return max(eigvals.min(), 0.0)
        except np.linalg.LinAlgError:
            return 0.0
    
    def required_noise_std(self):
        """Return noise standard deviation needed to maintain PE."""
        current_min = self.get_min_eigenvalue()
        if current_min >= self.lambda_min:
            return 0.0
        # Heuristic: required noise magnitude (simplified)
        return min(1.0, np.sqrt(max(0, self.lambda_min - current_min)))
