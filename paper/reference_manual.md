# Casanntra Manual (Reference)

*Note: This manual predates the MultiScenarioModelBuilder implementation*

---

## Glossary

| Term | Definition |
|------|------------|
| **Base** | The reference/source scenario (typically DSM2 or SCHISM contemporary Delta) |
| **CalSIM** | California water resources operations optimization model (monthly timestep) |
| **Case** | A distinct experimental configuration; cases partition time series to prevent cross-validation leakage |
| **Contrastive training** | Two-head architecture learning target, source, and their difference simultaneously |
| **Direct training** | Classic transfer learning; continue training on new data |
| **DSM2** | 1D Delta Simulation Model (fast, well-calibrated) |
| **GRU** | Gated Recurrent Unit (recurrent neural network layer) |
| **LSTM** | Long Short-Term Memory (recurrent neural network layer) |
| **Multi-scenario training** | Joint training across multiple scenarios with shared trunk |
| **RMA** | Resource Management Associates 2D/1D Bay-Delta model |
| **Scenario** | A specific configuration (SLR, restoration, barriers) |
| **SCHISM** | 3D Semi-implicit Cross-scale Hydroscience Integrated System Model |
| **Target** | The scenario being transferred to |
| **Transfer Learning** | Using a pre-trained model as starting point for new task |

---

## Scenarios

| ID | Description |
|----|-------------|
| DSM2 Base | Contemporary Delta, 1D model |
| SCHISM Base | Contemporary Delta, 3D model |
| SCHISM Suisun | Suisun Marsh restoration |
| SCHISM SLR | Sea level rise (1ft, 2.7ft) |
| SCHISM Cache | Cache Slough restoration |
| SCHISM Franks | Franks Tract Futures |

---

## Repository Structure

### 1. Run Configuration and Execution

**transfer_example.py** (entry point)
- Defines YAML config file and steps to run
- Calls `process_config(configfile, ["dsm2_base", ...])`
- Disables GPUs by default

**YAML configuration**
- Defines: output_dir, inputs, stations (outputs), ndays
- Training steps: dsm2_base → dsm2.schism → base.suisun
- Per-step hyperparameters: epochs, learning rates, etc.

**staged_learning.py** (orchestration)
- `process_config()`: reads YAML, instantiates builder, iterates steps
- `fit_from_config()`: loads data, creates folds, runs xvalid, saves model

### 2. Data Processing

**read_data.py**
- Finds files matching prefix pattern
- Optional regex mask to exclude datasets
- Required columns: `datetime`, `case`, all features, all stations

**Lagged sequences**
- GRUs expect shape `(batch, ndays, n_features)`
- `create_antecedent_inputs()` builds windows of previous ndays
- Example: batch=64, ndays=105, features=8 → `(64, 105, 8)`

### 3. Cross-Validation

**xvalid.py** (single output)
- Writes reference files before training
- Splits dataset into folds by datetime/case
- Builds model, fits, predicts withheld fold

**xvalid_multi.py** (multi-output, current default)
- Supports: single, two-head (contrastive), multi-scenario
- Parallel fold runs with `ProcessPoolExecutor`
- Multiple head predictions: `{output_prefix}_xvalid_{i}.csv`
  - i=0: target/scenario head
  - i=1: base head

**Output files**

| Training Type | File | Contents |
|---------------|------|----------|
| Direct | `_xvalid.csv` | Out-of-fold predictions |
| Direct | `_xvalid_ref_out_unscaled.csv` | Reference data |
| Contrastive | `_xvalid_0.csv` | Target predictions |
| Contrastive | `_xvalid_1.csv` | Base predictions |
| Contrastive | `_xvalid_ref_out_unscaled.csv` | Target reference |
| Contrastive | `_xvalid_ref_out_secondary_unscaled.csv` | Base reference |

### 4. Model and Training Details

**Data shape pipeline**
1. Raw data → `create_antecedent_inputs()` → lagged columns
2. `prepro_layers()` → per-feature scaling:
   - Rescaling(1.0) for gates (dcc, smscg)
   - ModifiedExponentialDecayLayer for flows (compress high values)
   - Normalization for other features
3. Stack → `(batch, ndays, n_features)`

**build_model()**
1. Preprocess → Reshape → Concatenate
2. Build architecture from `feature_layers` (GRU/LSTM stack)
3. Create heads:
   - Direct: single Dense head
   - Contrastive: two Dense heads + Subtract tensor

**Direct vs Contrastive**

| Aspect | Direct | Contrastive |
|--------|--------|-------------|
| Heads | 1 (target) | 2 (target + base) + contrast |
| Output | y_actual | y_target, y_base, (y_target - y_base) |
| Loss | MAE | target + base + λ·contrast |
| Weight init | From previous step | Both heads from previous |

**Scaling & Masking**
- `ScaledMaskedMAE`: per-station scale factors
- NaN handling: replace with predictions (zero gradient)

### 5. Outputs & Logs

**Model files**
- `{save_model_fname}.h5` → model graph + custom layers
- `{save_model_fname}.weights.h5` → trained weights
- Both required for transfer learning!

### 6. YAML Cheat Sheet

```yaml
output_dir: "./output"

model_builder_config:
  builder_name: MultiStageModelBuilder
  args:
    input_names: [northern_flow, exports, sjr_flow, ...]
    output_names: {x2: 100.0, pct: 20000.0, ...}  # station: scale
    ndays: 105

steps:
  - name: dsm2_base
    input_prefix: dsm2_base
    output_prefix: "{output_dir}/dsm2_base_gru2"
    save_model_fname: "{output_dir}/dsm2_base_gru2"
    load_model_fname: null  # First step, no prior model

    target_fold_length: 180d
    pool_size: 12
    pool_aggregation: true

    init_train_rate: 0.008
    init_epochs: 10
    main_train_rate: 0.001
    main_epochs: 35

    builder_args:
      transfer_type: direct  # or contrastive
      feature_layers:
        - {type: GRU, units: 32, name: lay1, trainable: true, return_sequences: true}
        - {type: GRU, units: 16, name: lay2, trainable: true, return_sequences: false}
```

### 7. Workflow of a Single Run

```
transfer_example.py
    │
    └─→ process_config(yaml, steps)
            │
            ├─→ read_config() → load YAML
            │
            ├─→ model_builder_from_config() → instantiate builder
            │
            └─→ fit_from_config() [per step]
                    │
                    ├─→ read_data() → load CSVs
                    │
                    ├─→ set load_model_fname for weight reuse
                    │
                    ├─→ xvalid_time_folds() → create splits
                    │
                    ├─→ write_reference_outputs() → ref CSVs
                    │
                    ├─→ xvalid_fit_multi() [per fold, parallel]
                    │       │
                    │       ├─→ create lagged inputs
                    │       ├─→ build_model()
                    │       ├─→ fit_model()
                    │       └─→ predict & write fold results
                    │
                    └─→ bulk_fit() → final training, save .h5
```
