# Regime-Aware Portfolio Management via Retrieval-Augmented LLM-Guided Expert Switching

Implementation of the framework described in the paper. Modular pipeline: data loading, VAE embedding, offline indexing with LLM-generated SoP documents, online retrieval and expert switching, and backtesting.

## Project Structure

```
├── config/experiment/          # Experiment override configs (K, λ, LLM model)
├── scripts/
│   ├── download_data.py        # Fetch raw market data
│   ├── run_backtest.py         # Single backtest (Tables 3–6)
│   ├── run_experiments.py      # Multi-run sensitivity analysis harness
│   ├── run_indexing.py         # Offline indexing pipeline
│   └── run_inference.py        # Online inference
├── src/                        # Core library
│   ├── backtest/               # Backtesting engine & metrics
│   ├── config/                 # Hydra base config
│   ├── data/                   # Data loading & feature engineering
│   ├── embedding/              # VAE & embedding utilities
│   ├── experts/                # DRL expert implementations
│   ├── indexing/               # FAISS + SQLite vector database
│   ├── inference/              # Online retrieval & switching
│   └── utils/                  # Helpers, logging, seeding
├── results/                    # Output CSVs
├── plots/                      # Generated figures
└── run.py                      # Hydra entry point
```

## Running Experiments

### Sensitivity Analysis Harness

The experiment harness systematically varies key hyperparameters (K, λ, LLM model) and aggregates results:

```bash
# Run all 9 K × λ combinations
python scripts/run_experiments.py

# Run specific experiments
python scripts/run_experiments.py --experiments K5_lambda0.1 K10_lambda0.5

# Override LLM model for all experiments
python scripts/run_experiments.py --llm-model gpt-4

# Dry-run (validate configs only)
python scripts/run_experiments.py --dry-run
```

### Experiment Configs

Located in `src/config/experiment/`:

| Config file               | K  | λ   |
|---------------------------|----|-----|
| `K5_lambda0.1.yaml`       | 5  | 0.1 |
| `K5_lambda0.5.yaml`       | 5  | 0.5 |
| `K5_lambda1.0.yaml`       | 5  | 1.0 |
| `K10_lambda0.1.yaml`      | 10 | 0.1 |
| `K10_lambda0.5.yaml`      | 10 | 0.5 |
| `K10_lambda1.0.yaml`      | 10 | 1.0 |
| `K20_lambda0.1.yaml`      | 20 | 0.1 |
| `K20_lambda0.5.yaml`      | 20 | 0.5 |
| `K20_lambda1.0.yaml`      | 20 | 1.0 |

LLM model and embedding dimension overrides:

| Config file               | Override                          |
|---------------------------|-----------------------------------|
| `llm_gpt35.yaml`          | `llm.model_name=gpt-3.5-turbo`    |
| `llm_gpt4.yaml`           | `llm.model_name=gpt-4`            |
| `llm_local.yaml`          | `llm.model_name=local-model`      |
| `embed_dim16.yaml`        | `latent_dim_tech=16, latent_dim_mkt=16` |
| `embed_dim32.yaml`        | `latent_dim_tech=32, latent_dim_mkt=32` |

### Outputs

```
results/experiments/
├── K5_lambda0.1/
│   ├── metrics.csv           # Per-experiment metrics
│   └── config.yaml           # Copy of exact config used
├── K5_lambda0.5/
│   ...
├── summary.csv               # All experiments combined
├── summary_statistics.csv    # Mean ± std per strategy
└── sensitivity_heatmap.png   # Sharpe by K × λ
plots/experiments/
├── sensitivity_heatmap.png
├── sensitivity_lines.png
└── llm_sensitivity.png
```

### Single Backtest Run
 
 ```bash
 python scripts/run_backtest.py
 python scripts/run_backtest.py --mock-llm --no-plots
 ```
 
 ### Data Fetching
 
 Download raw data for Crypto, NASDAQ, and Forex:
 
 ```bash
 export PYTHONPATH=$PYTHONPATH:. && python scripts/download_data.py --start 2024-01-01 --end 2024-03-01
 ```
 
 ### Data Visualization
 
 Generate OHLC charts for all downloaded symbols:
 
 ```bash
 export PYTHONPATH=$PYTHONPATH:. && python scripts/visualize_raw_data.py
 ```
 
 ### Hydra Multirun (Alternative)


```bash
python run.py retrieval.K=5,10,20 uncertainty.lambda=0.1,0.5,1.0 --multirun
```

## Dependencies

See `pyproject.toml`. Key: hydra-core, omegaconf, torch, faiss-cpu, litellm, stable-baselines3.
