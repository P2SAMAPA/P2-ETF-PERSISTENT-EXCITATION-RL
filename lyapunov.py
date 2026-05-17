import numpy as np
from config import LYAPUNOV_BETA, LYAPUNOV_THRESHOLD

class LyapunovMonitor:
    """
    Simple Lyapunov function: V = 0.5 * ||θ - θ*||^2 (if we had true θ*).
    We approximate using parameter change magnitude.
    """
    def __init__(self, beta=LYAPUNOV_BETA, threshold=LYAPUNOV_THRESHOLD):
        self.beta = beta
        self.threshold = threshold
        self.prev_params = None
        self.V = 0.0
    
    def update(self, current_params):
        """Compute Lyapunov decrement and check stability."""
        if self.prev_params is None:
            self.prev_params = current_params.copy()
            self.V = 0.0
            return True, 0.0
        
        diff = np.linalg.norm(current_params - self.prev_params)
        # Lyapunov decrease condition: V_new - V_old <= -beta * diff^2
        V_new = 0.5 * diff**2
        delta_V = V_new - self.V
        stable = (delta_V <= -self.beta * diff**2) or (V_new < self.threshold)
        self.V = V_new
        self.prev_params = current_params.copy()
        return stable, delta_V
