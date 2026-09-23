"""
Demo Script for Secure Analytics Engine
Simulates baseline operational data vs tampered/poisoned data stream.
"""

import numpy as np
import pandas as pd
from engine import RuntimeIntegrityMonitor

def main():
    print("=== Secure Analytics Engine: Demo ===")
    np.random.seed(42)

    # 1. Generate normal operational baseline data (e.g., sensor or embeddings feature stream)
    clean_data = pd.DataFrame({
        "feature_a": np.random.normal(loc=10.0, scale=2.0, size=1000),
        "feature_b": np.random.normal(loc=50.0, scale=5.0, size=1000),
        "feature_c": np.random.normal(loc=0.0, scale=1.0, size=1000)
    })

    # Initialize and fit monitor
    monitor = RuntimeIntegrityMonitor(z_threshold=3.0, IQR_multiplier=1.5)
    monitor.fit_baseline(clean_data)

    # Test baseline integrity
    clean_report = monitor.evaluate_integrity(clean_data)
    print("\n[Baseline Data Integrity Check]:")
    print(clean_report)

    # 2. Simulate adversarial data injection (tampering)
    corrupted_data = clean_data.copy()
    # Inject extreme outliers into 50 rows of feature_a and feature_b
    corrupted_data.loc[100:150, "feature_a"] += 25.0
    corrupted_data.loc[200:220, "feature_b"] -= 40.0

    # Evaluate corrupted dataset
    corrupted_report = monitor.evaluate_integrity(corrupted_data)
    print("\n[Tampered Data Stream Integrity Check]:")
    print(corrupted_report)

if __name__ == "__main__":
    main()
