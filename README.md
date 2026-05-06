# Reinforcement Learning Trading Environment

Hourly BTCUSDT trading agent trained with Stable Baselines 3 using a semi-Markov
formulation of the sequential decision problem. Walk-forward validation across four
folds from 2017 to 2026.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python code/main.py
```
