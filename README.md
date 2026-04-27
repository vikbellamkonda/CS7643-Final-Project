# CS7643-Final-Project
Time Series Project for Deep Learning

This repository contains the core data pipeline for our volatility forecasting project. 

## Quick Start (For Will & Arjun)
To use this data in your model scripts, use the following snippet:

```python
from test_data import prepare_mist_data, MISTDataset
from torch.utils.data import DataLoader

# 1. Load Data (Auto-caches to .pkl)
returns, targets = prepare_mist_data()

# 2. Setup Dataset (lookback=10 or 60)
dataset = MISTDataset(returns, targets, lookback=60)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. Train
for x, s_id, y in loader:
    # x: [Batch, 60, 1] | s_id: [Batch, 99] | y: [Batch]
    pass
