# Paper Outline

## Working Title
**Transfer Learning for Multi-Fidelity Surrogate Modeling of Delta Salinity Under Extended Drought and Climate Scenarios**

---

## Project Context (from DSC Pilot Project Scope)

### Challenge
- California experiencing historic drought; climate change will intensify
- Sacramento-San Joaquin Delta is critical for fresh water supply
- Net Delta outflow must counter tidal mixing of ocean salt
- "Water cost" = freshwater volume needed to maintain salinity standards
- During extended drought, water cost is significant relative to reservoir storage

### Approach
- Integrate multi-dimensional hydrodynamic models with CalSim operations model
- Use transfer learning to create surrogate ANNs for novel scenarios
- Evaluate drought mitigation strategies (landscape changes, barriers, SLR)

### Key Innovation
- Transfer learning bridges the gap between:
  - Computationally cheap DSM2 (1D, 100-year training sets)
  - Expensive SCHISM/RMA (3D/2D, limited scenario runs)
- Multi-scenario architecture enables joint training across interventions

---

## Section-by-Section Notes

### 1. Introduction
- Delta as critical water infrastructure
- Drought intensification under climate change
- D-1641 standards and the concept of water cost
- Gap: Current ANNs trained on DSM2 baseline only
- Solution: Transfer learning from coarse to fine models
- Paper contribution: casanntra framework + multi-scenario architecture

### 2. Background

#### 2.1 Sacramento-San Joaquin Delta
- Geography: confluence of Sacramento and San Joaquin rivers
- Tidal influence from San Francisco Bay
- Salt intrusion mechanisms (gravitational circulation + tidal mixing)
- X2 as salinity indicator (distance to 2 psu isohaline)
- Compliance locations and D-1641 requirements

#### 2.2 Modeling Hierarchy
- **DSM2**: 1D cross-sectional model, fast, well-calibrated for contemporary Delta
- **RMA Bay-Delta**: 2D/1D hybrid, captures lateral variation
- **SCHISM**: 3D, resolves vertical stratification (critical for SLR)
- **CalSim**: Monthly optimization model, 80+ year simulations
- Current state: CalSim uses ANN trained from DSM2 only

#### 2.3 Surrogate Models
- G-model concept: salinity as function of antecedent outflow
- ANN architecture evolution
- Limitation: novel scenarios (SLR, restoration) not in DSM2 training

### 3. Methods

#### 3.1 Transfer Learning Framework
- **Staged training pipeline**:
  1. DSM2 base (extensive data, learn general salinity-flow relationships)
  2. SCHISM base (limited data, adapt to 3D physics)
  3. Scenario variants (contrastive learning for deltas)
- Weight initialization from previous stage
- Freeze schedules to preserve learned features
- **Direct transfer**: continue training on new data (classic approach)
- **Contrastive transfer**: learn target, source, and difference simultaneously

#### 3.2 Contrastive Loss Formulation
- Multi-head architecture: source head, target head, contrast head
- Loss = L(source) + L(target) + λ·L(contrast)
- Contrast = target - source (learns the delta directly)
- Both heads initialized from previous step's weights
- Contrast head computed internally (not written as output)

#### 3.3 Multi-Scenario Architecture
- Shared trunk (GRU layers) across all scenarios
- Scenario-specific heads
- Optional per-scenario branch layers
- Enables: base + suisun + slr + cache + franks in single model
- **Parity constraint**: single-scenario MSCEN must match MSTAGE results

#### 3.4 Data Pipeline
- **Case structure**: partitions time series into independent experiments
  - Cases never split during cross-validation
  - Prevents data leakage across experimental conditions
- **Antecedent inputs**: create_antecedent_inputs() builds lagged windows
  - Each feature expanded to ndays columns (e.g., input_lag0...input_lag104)
  - Shape: (batch, ndays, n_features) for GRU input
- **Feature preprocessing** (prepro_layers):
  - Gates (dcc, smscg): Rescaling(1.0) - identity
  - Flows (northern_flow, sjr_flow): ModifiedExponentialDecayLayer
    - Compresses high flows that don't affect salinity
    - Parameter b=70000 for northern, b=40000 for SJR
  - Other features: Normalization (zero mean, unit variance)

#### 3.5 Neural Network Architecture
- Input: 8 features × ndays antecedent (90-120 days)
  - northern_flow, exports, sjr_flow, cu_delta
  - sf_tidal_energy, sf_tidal_filter, dcc, smscg
- Preprocessing → Reshape → Concatenate → (batch, ndays, 8)
- Trunk: 2 stacked GRUs (32→16 or 38→19 units)
  - lay1: return_sequences=True
  - lay2: return_sequences=False (time-collapsed output)
- Heads: Dense layer per output station (19 stations)
- Activation: ELU
- Output: scaled salinity predictions

#### 3.6 Training Details
- **Loss**: ScaledMaskedMAE
  - Per-station scale factors (e.g., x2: 100, pct: 20000)
  - NaN masking: replace missing with predictions (zero gradient contribution)
- **Optimizer**: Adamax with gradient clipping (clipnorm=1.0)
- **Two-phase training**:
  - Init phase: high LR (0.008), 10 epochs, run_eagerly=True
  - Main phase: low LR (0.001), 35 epochs, run_eagerly=False
- **Cross-validation**:
  - Time-based folds (target_fold_length: 180d)
  - Case boundaries preserved
  - Parallel fold execution (pool_size workers)

#### 3.7 Scenarios
| ID | Description | Model Source |
|----|-------------|--------------|
| base | Contemporary Delta | DSM2/SCHISM |
| suisun | Suisun Marsh restoration | SCHISM |
| slr | 1ft / 2.7ft sea level rise | SCHISM |
| cache | Cache Slough restoration | SCHISM |
| franks | Franks Tract Futures | SCHISM/RMA |

### 4. Results

#### 4.1 Transfer Learning Performance
- DSM2 → SCHISM base: NSE improvement metrics
- Table: Station-by-station NSE comparison
- Figure: Learning curves showing transfer benefit

#### 4.2 Hyperparameter Sensitivity
- Grid search over: ndays, layer sizes, freeze schedules, contrast weight
- Best configuration: 38→19 GRU, freeze=[0,0,1], ndays=105

#### 4.3 Multi-Scenario Parity
- Single-scenario (MSTAGE) vs multi-scenario (MSCEN) comparison
- Demonstrate statistical equivalence when configured identically
- Computational efficiency of joint training

#### 4.4 Round-Trip Validation
- CalSim runs with updated ANNs
- Multi-dimensional model validation of CalSim outputs
- Water cost estimates for drought mitigation scenarios

### 5. Discussion
- Benefits of transfer learning for limited-data scenarios
- Multi-scenario efficiency for exploring interventions
- Limitations:
  - Catastrophic drought not modeled
  - Climate predictions not included
  - Assumes D-1641-like compliance framework
- Future: Embed SLR as continuous parameter, expand scenario library

### 6. Conclusions
- Transfer learning successfully bridges DSM2→SCHISM gap
- Multi-scenario architecture enables efficient joint training
- Framework ready for operational integration with CalSim
- Open-source: casanntra library

---

## Figures Needed
1. Delta map with compliance stations
2. Modeling workflow diagram (from scope Fig. 1)
3. Neural network architecture diagram
4. Transfer learning pipeline (3 stages)
5. Multi-scenario architecture (trunk + heads)
6. Learning curves (DSM2 → SCHISM → scenario)
7. Station-by-station NSE comparison
8. Hyperparameter sensitivity heatmaps
9. MSTAGE vs MSCEN parity scatter plot
10. Round-trip validation timeseries

## Tables Needed
1. Model comparison (DSM2/RMA/SCHISM/CalSim)
2. Input features and scaling
3. Output stations and scale factors
4. Hyperparameter grid and results
5. Station-level metrics for best configuration
6. Water cost comparison across scenarios

---

## Status
- [ ] Introduction
- [ ] Background
- [ ] Methods
- [ ] Results
- [ ] Discussion
- [ ] Conclusions
- [ ] Figures
- [ ] Tables
