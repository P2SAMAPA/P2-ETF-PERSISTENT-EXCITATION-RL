import numpy as np
from collections import deque

class PersistentExcitation:
    """
    Maintains a rolling covariance matrix of feature vectors (e.g., state-action pairs)
    and adjusts exploration noise to keep the smallest eigenvalue above lambda_min.
    """
    def __init__(self, feature_dim, lambda_min=0.01, window=1000):
        self.feature_dim = feature_dim
        self.lambda_min = lambda_min
        self.window = window
        self.buffer = deque(maxlen=window)
        self.cov = np.eye(feature_dim) * 1e-6
        self.mean = np.zeros(feature_dim)
        self.n = 0
    
    def update(self, feature_vector):
        """Add new feature vector (e.g., concatenated state and action)."""
        self.buffer.append(feature_vector)
        # Incremental covariance update
        if self.n == 0:
            self.mean = feature_vector.copy()
            self.cov = np.zeros((self.feature_dim, self.feature_dim))
        else:
            old_mean = self.mean.copy()
            self.mean = (self.mean * self.n + feature_vector) / (self.n + 1)
            self.cov = ((self.cov * self.n) + 
                        np.outer(feature_vector - old_mean, feature_vector - self.mean))
        self.n = min(self.n + 1, self.window)
    
    def get_min_eigenvalue(self):
        if self.n < self.feature_dim:
            return 0.0
        # Use sample covariance (unbiased)
        sample_cov = self.cov / (self.n - 1) if self.n > 1 else self.cov
        eigvals = np.linalg.eigvalsh(sample_cov)
        return np.maximum(eigvals.min(), 0.0)
    
    def required_noise_std(self):
        """Return noise standard deviation needed to keep lambda_min >= target."""
        current_min = self.get_min_eigenvalue()
        if current_min >= self.lambda_min:
            return 0.0
        # Heuristic: increase noise to raise eigenvalue (very simplified)
        # In practice, one would add noise to actions/parameters.
        return np.sqrt(max(0, self.lambda_min - current_min))
