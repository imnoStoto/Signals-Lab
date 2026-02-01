"""
Plots datapoints for mit-bih arrhythmia set.
"""


import wfdb
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    record_path = Path(
        "assets/datasets/raw/mit-bih/mit-bih-arrhythmia-database-1.0.0/100"
    )

    signals, fields = wfdb.rdsamp(str(record_path))

    plt.plot(signals[:2000, 0])
    plt.title("ECG waveform (channel 0)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()

if __name__ == "__main__":
    main()
