# HRIT v3 — Computational Implementation Suite

**Hierarchical Recursive Integration Theory (HRIT v3)**  
*Interoceptive Depth and Integratation as Independent Axes of Conscious Experience*

**Author:** Hamza S. Almahi · hamza.s.almahi@gmail.com  
**Preprint:** [doi.org/10.5281/zenodo.19490741](https://doi.org/10.5281/zenodo.19490741)  
**Date:** May 2026

---

## Overview

This repository contains the complete active inference implementation of the HRIT v3 computational architecture, implemented in MATLAB using SPM25 (`spm_MDP_VB_X`). All four components are confirmed working and produce the predicted clinical signatures. Clinical state simulations run from first principles — no manually specified values.

The repository demonstrates that the HRIT v3 architecture is computationally realisable in the active inference framework (Friston, 2010; Smith, Friston & Whyte, 2022; Whyte et al., 2026), and generates a fully simulated Consciousness State Vector (CSV) state space across seven clinical conditions.

---

## Three-Layer Terminology (Locked)

| Phenomenological | Pre-computational | Computational | Variable |
|---|---|---|---|
| Simulation | Hierarchical Perceptual Inference | Two-level POMDP (spm_MDP_VB_X) | `I_Sim` |
| Identity | Autobiographical Self-Modelling | Deep Temporal Model (slow B, concentrated D) | `I_Id` |
| Awareness | Metacognitive Monitoring | Second-order Precision Control POMDP | `I_Prec` |
| Vitality | Allostatic Interoceptive Regulation | Signed Predictive Coding ODE | `I_Intero → PDS` |

---

## Requirements

- MATLAB R2020b or later
- SPM25: [www.fil.ion.ucl.ac.uk/spm](https://www.fil.ion.ucl.ac.uk/spm)
- SPM25 DEM toolbox must be on the MATLAB path:
  ```matlab
  addpath('path/to/spm/toolbox/DEM')
  ```
- Confirm SPM25 is working before running any script:
  ```matlab
  which spm_MDP_VB_X
  % Should return a valid path
  ```

No additional toolboxes required. All scripts are self-contained.

---

## Repository Structure

```
HRIT-v3-computational/
│
├── README.md                    % This file
│
├── HRIT_I_Intero_v4.m           % I_Intero: Allostatic Interoceptive Regulation
│                                %   v4: six physiological states (healthy regulation)
│                                %   Sleep onset → ICP drop ~28% (matches Study 1c)
│
├── HRIT_I_Intero_v3.m           % I_Intero v3: three clinical pathological states
│                                %   (Normal, DPDR, Cotard) — retained for reference
│
├── HRIT_I_Sim_v2.m              % I_Sim: Hierarchical Perceptual Inference (two-level POMDP)
├── HRIT_I_Id_v1.m               % I_Id: Autobiographical Self-Modelling
├── HRIT_I_Prec_v1.m             % I_Prec: Metacognitive Monitoring
│
└── HRIT_Integration_v2.m        % Full integration: CII, p(Ψ), PDS, CSV state space
```

---

## Scripts

### `HRIT_I_Intero_v4.m` — Vitality / Allostatic Interoceptive Regulation *(primary)*

Implements the signed predictive coding ODE for interoceptive inference. Updated from v3 to model **six physiological regulatory states** in the healthy brain. v3 modelled clinical pathologies; v4 models the full adaptive range of normal allostatic regulation.

**Key equations (unchanged from v3):**
```
dμ_intero/dt = ε_o/σ_o + ε_p/σ_p
a = −κ · ε_o / σ_o  (scaled by action efficacy in sleep)
F = 0.5·(ε_o²/σ_o + ε_p²/σ_p)
ICP = 1 / (1 + |ε_o|)
PDS = tanh[(w·(ICP − ICP_ref) − β·CAL) / PSC]
PE_residual = |dF/dt|
Allostatic Reserve = PDS_max_running − PDS_current
```

**v4 additions:**
- Time-varying σ_o per condition (precision changes within a condition)
- Time-varying u(t) stress input (function handle per condition)
- Action efficacy parameter (0.70 at N3 — interoceptive decoupling during sleep)
- Sleep reset events: CAL ← CAL × (1 − ρ_sleep), ρ_sleep = 0.50
- Allostatic Reserve output: regulatory capacity remaining

**Six physiological states:**

| Condition | σ_o | κ | Efficacy | Sleep reset | PDS (ss) | ICP (ss) |
|---|---|---|---|---|---|---|
| Waking Rest | 1.0 | 0.50 | 1.00 | — | **+0.989** | 1.000 |
| Physical Exertion + Recovery | 0.5 | 0.70 | 1.00 | ✓ | **+0.989** | 1.000 |
| Psychological Stress + Recovery | 1.2 | 0.50 | 1.00 | ✓ | **+0.989** | 1.000 |
| Sleep Onset (Wake → N3) | 1.0→3.0 | 0.50 | 1.0→0.7 | ✓ | **+0.15–0.30** | ~0.72 |
| Flow / Optimal Regulation | 0.5 | 0.90 | 1.00 | — | **+0.999** | 1.000 |
| Allostatic Overload (no reset) | 1.5 | 0.45 | 1.00 | — | **falling** | falling |

**Connection to Study 1c (Sleep EEG):**
The Sleep Onset condition reproduces the empirical finding from Almahi (2026):
- Model: ICP drop ~28–30% at N3, PE_residual change ~5% → ratio ~5–6:1
- Study 1c: HEP amplitude drop 27.7%, PE drop 5.4% → ratio 5.1:1
This establishes the sleep N3 state as the physiological reference range for interoceptive precision in the HRIT framework.

**Run:** `HRIT_I_Intero_v4` — produces 3 figures and command window summary including Study 1c comparison.

---

### `HRIT_I_Intero_v3.m` — Vitality / Allostatic Interoceptive Regulation *(clinical reference)*

Retained for backward compatibility. Models three clinical pathological states: Normal, DPDR (precision collapse), and Cotard (depth inversion). See v4 for physiological extension.

**Run:** `HRIT_I_Intero_v3` — produces 3 figures and command window summary.

---

### `HRIT_I_Sim_v2.m` — Simulation / Hierarchical Perceptual Inference

True two-level hierarchical POMDP using `MDP.link = 1` to pass Level-1 posteriors (tone identity; 2 states, 4 timesteps) to Level-2 (sequence pattern; 4 states, 6 trials). Reproduces MMN (Level-1 PE) and P3b (Level-2 PE) equivalents.

**Architecture:**
- Level 1: infers tone identity per timestep (high/low)
- Level 2: infers global sequence pattern across trials (HHHL, LLLH, HHHH, LLLL)
- Link: `MDP.link = 1` — Level-1 posteriors become Level-2 observations

**Key extractor (confirmed):**
```matlab
I_Sim       = abs(acc_baseline) / abs(acc_current)
PE_residual = norm(MDP.vn{1}(end,:))
MC_Sim      = mean(-MDP.F) - abs(acc)
```

**Confirmed results:**

| Condition | I_Sim |
|---|---|
| Normal Waking | **1.000** (reference) |
| Integration Collapse (A=0.20, B=0.20) | **0.143** |
| Learning Flow (concentrated D2 prior) | **1.146** |

**Run:** `HRIT_I_Sim_v2` — produces 2 figures and results table.

---

### `HRIT_I_Id_v1.m` — Identity / Autobiographical Self-Modelling

Deep Temporal POMDP with slow B matrix (B_stab = 0.99 Normal, 0.50 Degraded) and high-concentration D vector prior. Captures autobiographical self-continuity through slow-timescale prior stability. T = 8 timesteps (longer than I_Sim to capture temporal depth).

**Confirmed results:**

| Condition | I_Id |
|---|---|
| Normal | **1.000** (reference) |
| Degraded (B=0.50, flat D) | **0.347** |
| Enhanced (B=0.999, concentrated D) | **1.062** |

**Run:** `HRIT_I_Id_v1` — produces 2 figures and results table.

---

### `HRIT_I_Prec_v1.m` — Awareness / Metacognitive Monitoring

Second-order Precision Control POMDP. A-matrix precision at the meta-awareness level controls accuracy of second-order state inference. Behavioural correlate: meta-d′/d′. T = 6 timesteps.

**Confirmed results:**

| Condition | I_Prec |
|---|---|
| Normal (A2=0.90) | **1.000** (reference) |
| Degraded (A2=0.55, flat D) | **0.254** |
| Enhanced (A2=0.98, concentrated D) | **4.361** |

**Run:** `HRIT_I_Prec_v1` — produces 2 figures and results table.

---

### `HRIT_Integration_v2.m` — Full CSV State Space

Runs all four components simultaneously under matched parameter sets for seven clinical states. All values derived from simulation — no manually specified component values. Computes CII, p(Ψ), PDS_expressed, and CSV for each state.

**Core equations:**
```
CII(Ψ) = [0.40·I_Sim^p + 0.35·I_Id^p + 0.25·I_Prec^p]^(1/p)

p(Ψ)   = PE_residual_dominant / (MC_dominant + ε)
         dominant = argmax(MC_Sim, MC_SelfRef)
         MC_SelfRef = MC_Id + MC_Prec
         PE_residual = |dF/dt| at near-equilibrium

PDS_expressed(Ψ) = PDS(Ψ) · σ(k·(CII(Ψ) − MPT))
MPT(t,age,h)     = α · CII_max(t,age,h)

CSV(Ψ) = (CII, PDS_expressed) ∈ [0,∞) × [−1,+1]
```

**Fully simulated CSV state space (CII_max = 1.920, MPT = 0.288, α = 0.15):**

| State | I_Sim | I_Id | I_Prec | p(Ψ) | CII | PDS_expressed |
|---|---|---|---|---|---|---|
| Normal Waking | 1.000 | 1.000 | 1.000 | 0.570 | **1.000** | +0.962 |
| DPDR (Depth Collapse) | 1.000 | 1.000 | 0.255 | 0.326 | **0.749** | +0.152 |
| Cotard (Depth Inversion) | 1.000 | 0.347 | 0.255 | 0.005 | **0.491** | −0.734 |
| Propofol Anaesthesia | 0.143 | 0.347 | 0.255 | 0.005 | **0.225** | +0.418 |
| Expert Flow | 0.991 | 1.062 | 4.361 | 1.406 | **2.058** | +0.989 |
| Learning Flow | 1.146 | 0.955 | 1.267 | 0.422 | **1.105** | +0.973 |
| Anxiety/Fragmented | 0.667 | 0.446 | 0.255 | 0.009 | **0.456** | +0.685 |

**CII ordering:** Expert Flow (2.058) > Learning Flow (1.105) > Normal Waking (1.000) ✓  
**PDS:** Normal (+0.962), DPDR near zero (+0.152), Cotard at negative attractor (−0.734) ✓

**Run:** `HRIT_Integration_v2` — produces 3 figures and full CSV table. Takes approximately 5–10 minutes.

---

## Key Computational Contributions

**1. Unified PE_residual definition**  
PE_residual = |dF/dt| at near-equilibrium unifies the continuous I_Intero model (where it is the valence signal magnitude) and the discrete POMDP models (where it is the norm of the state prediction error vector at convergence: `norm(MDP.vn{1}(end,:))`).

**2. Corrected p(Ψ) formula**  
p(Ψ) = PE_residual_dominant / (MC_dominant + ε) with range [0, ∞). The dominant subsystem (Simulation vs Self-Reference) is determined by which has higher metabolic integration cost at the current moment.

**3. True hierarchical I_Sim**  
Level-1 posteriors passed to Level-2 via `MDP.link = 1` following Smith, Friston & Whyte (2022) Section 6. Reproduces MMN (Level-1 PE) and P3b (Level-2 PE) equivalents from a single hierarchical model.

**4. Signed PDS with Cotard attractor**  
PDS is signed — crosses zero when ICP falls below ICP_ref. Stable negative attractor at −1 emerges from dual-condition mechanism: moderately reduced precision (σ_o = 5.0) AND sustained negative push (−0.5). Neither condition alone produces the attractor.

**5. Fully simulated CSV state space**  
No manually specified component values. All seven clinical states derived from running the four component models under matched parameter sets.

---

## Confirmed Results Summary

All scripts confirmed working in SPM25, MATLAB, May 2026.

```
I_Intero v4 (physiological): Waking Rest=+0.989, Sleep N3≈+0.20, Flow≈+0.999 ✓
I_Intero v3 (clinical):  Normal=+0.989, DPDR=+0.167, Cotard=-1.000  ✓
I_Sim:    Normal=1.000,  Degraded=0.143, Flow=1.146   ✓
I_Id:     Normal=1.000,  Degraded=0.347, Enhanced=1.062 ✓
I_Prec:   Normal=1.000,  Degraded=0.254, Enhanced=4.361 ✓
CII:      Expert Flow (2.058) > Learning Flow (1.105) > Normal (1.000) ✓
PDS:      Normal (+0.962), DPDR (+0.152), Cotard (-0.734) ✓
Sleep N3 ICP drop: model ~28%, Study 1c empirical 27.7% ✓
```

---

## References

- Friston, K. J. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Smith, R., Friston, K. J., & Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. *Journal of Mathematical Psychology*, 107, 102632.
- Tschantz, A., Barca, L., Maisto, D., Buckley, C. L., Seth, A. K., & Pezzulo, G. (2022). Simulating homeostatic, allostatic and goal-directed forms of interoceptive control using active inference. *bioRxiv*.
- Whyte, C. J., et al. (2026). On the minimal theory of consciousness implicit in active inference. *Physics of Life Reviews*, 56, 4–28.
- Almahi, H. S. (2026). Hierarchical Recursive Integration Theory (HRIT v3). *Zenodo*. https://doi.org/10.5281/zenodo.19490741

---

## Citation

If you use this code, please cite:

```
Almahi, H. S. (2026). HRIT v3 Computational Implementation Suite [Software].
GitHub. https://github.com/hamzasalmahi-lab/HRIT-v3-computational
```

---

## Contact

Hamza S. Almahi · hamza.s.almahi@gmail.com  
Independent Researcher · Cairo, Egypt
