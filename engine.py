import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class RuntimeIntegrityMonitor:
    """
    Monitors streaming or batch data for adversarial tampering, 
    outlier injection, and distribution drift using stochastic statistical metrics.
    """

    def __init__(self, z_threshold: float = 3.0, IQR_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.IQR_multiplier = IQR_multiplier
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def fit_baseline(self, df: pd.DataFrame) -> None:
        """Computes statistical baseline parameters from trusted reference data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.baseline_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
                "iqr": float(df[col].quantile(0.75) - df[col].quantile(0.25))
            }

    def detect_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifies values deviating beyond the defined Z-score threshold."""
        anomalies = pd.DataFrame(index=df.index)
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c in self.baseline_stats]

        for col in numeric_cols:
            mean = self.baseline_stats[col]["mean"]
            std = self.baseline_stats[col]["std"] + 1e-9  # Avoid division by zero
            z_scores = np.abs((df[col] - mean) / std)
            anomalies[col] = z_scores > self.z_threshold

        return anomalies

    def detect_tampering_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Uses Interquartile Range (IQR) bounds to flag localized data injection/tampering."""
        tampered_flags = pd.DataFrame(index=df.index)
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c in self.baseline_stats]

        for col in numeric_cols:
            q25 = self.baseline_stats[col]["q25"]
            q75 = self.baseline_stats[col]["q75"]
            iqr = self.baseline_stats[col]["iqr"]

            lower_bound = q25 - (self.IQR_multiplier * iqr)
            upper_bound = q75 + (self.IQR_multiplier * iqr)

            tampered_flags[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        return tampered_flags

    def evaluate_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs full integrity analysis and returns an epistemic risk summary.
        """
        if not self.baseline_stats:
            raise ValueError("Baseline not fitted. Call fit_baseline() with reference data first.")

        z_anomalies = self.detect_outliers_zscore(df)
        iqr_anomalies = self.detect_tampering_iqr(df)

        total_cells = df.select_dtypes(include=[np.number]).size
        z_anomaly_count = int(z_anomalies.sum().sum())
        iqr_anomaly_count = int(iqr_anomalies.sum().sum())

        corruption_rate = max(z_anomaly_count, iqr_anomaly_count) / max(total_cells, 1)

        # Epistemic risk classification based on statistical bounds
        if corruption_rate == 0:
            status = "HEALTHY"
            confidence = 0.95
        elif corruption_rate < 0.05:
            status = "MINOR_DRIFT_DETECTED"
            confidence = 0.85
        elif corruption_rate < 0.15:
            status = "MODERATE_TAMPERING_SUSPECTED"
            confidence = 0.75
        else:
            status = "CRITICAL_DATA_CORRUPTION"
            confidence = 0.60

        return {
            "status": status,
            "corruption_rate_pct": round(corruption_rate * 100, 2),
            "z_score_violations": z_anomaly_count,
            "iqr_violations": iqr_anomaly_count,
            "epistemic_confidence": confidence
        }
