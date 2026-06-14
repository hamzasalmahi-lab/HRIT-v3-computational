# HRIT v3 — Computational Implementation

MATLAB code for the four components of the Hierarchical Recursive
Integration Theory (HRIT v3), plus the integration layer that produces
the two-dimensional Consciousness State Vector. Implemented in SPM25
(`spm_MDP_VB_X`) for the POMDP pieces, in plain MATLAB ODEs for the
allostatic component.

The theory is in the preprint:
[doi.org/10.5281/zenodo.19490741](https://doi.org/10.5281/zenodo.19490741).
This repository is the implementation — everything runs from first principles,
no hard-coded outputs.

## Three layers, four components

HRIT v3 keeps a strict separation between phenomenological description,
computational mechanism, and the variable that gets numerically simulated:

| Phenomenological | Computational mechanism | Variable |
|---|---|---|
| Simulation | Hierarchical Perceptual Inference (two-level POMDP) | `I_Sim` |
| Identity | Autobiographical Self-Modelling (slow B, concentrated D) | `I_Id` |
| Awareness | Metacognitive Monitoring (second-order precision POMDP) | `I_Prec` |
| Vitality | Allostatic Interoceptive Regulation (signed predictive coding ODE) | `I_Intero → PDS` |

Integration combines them into the Consciousness Integration Index (CII)
and Phenomenal Depth Signal (PDS), which span the CSV state space.

## Requirements

MATLAB R2020b or later, plus SPM25:

```matlab
addpath('path/to/spm/toolbox/DEM')
which spm_MDP_VB_X     % should return a path
```

Beyond SPM, no extra toolboxes are needed. Scripts are self-contained.

## Files

```
HRIT_I_Intero_v4.m       Allostatic regulation, six physiological states
HRIT_I_Intero_v3.m       Earlier version: three clinical states (DPDR, Cotard, normal)
HRIT_I_Sim_v3.m          Hierarchical perceptual inference, six physiological states
HRIT_I_Id_v2.m           Autobiographical self-modelling, six physiological states
HRIT_I_Prec_v2.m         Metacognitive monitoring, six physiological states
HRIT_Integration_v3.m    Full integration → CSV state space

HRIT_I_Sim_v2.m,         Earlier (clinical) versions of the POMDP components,
HRIT_I_Id_v1.m,          retained for reference
HRIT_I_Prec_v1.m,
HRIT_Integration_v2.m
```

## What changed in v3/v4

The earlier components modelled clinical pathological states (normal,
DPDR, Cotard, anaesthesia, etc.) — useful for the preprint, but the
range of interesting healthy dynamics was missing. The new versions add
six physiological states:

1. Waking rest (baseline)
2. Physical exertion + recovery
3. Psychological stress + recovery
4. Sleep onset (Wake → N3)
5. Flow / optimal regulation
6. Allostatic overload without sleep reset (pre-pathological)

The sleep onset condition is the one I'm most pleased with. With
`u_sleep = 0.045` (sustained slow oscillations during N3) and
`efficacy_N3 = 0.70` (motor atonia attenuating afferent coupling), the
model produces a 27.8% ICP drop from waking baseline to N3 — within
rounding of the 27.7% HEP amplitude drop in the Sleep-EDF Expanded dataset
(Studies 1 and 1c). The other physiological conditions hit their
qualitative targets: Flow gives PDS ≈ 1.0, Overload drives PDS negative
without any explicit pathology being modelled.

The POMDP components (`I_Sim_v3`, `I_Id_v2`, `I_Prec_v2`) inherit the same
six-state structure. Exact magnitudes need MATLAB/SPM25 to confirm — Python
approximation in `compare_models.py`-style scripts gives the right
ordering (Flow > Rest > Exertion > Stress > Overload > N3) but not the
absolute values.

## Connection to the FND models

HRIT v3 is the broader framework; the three FND-specific models —
[`fnd-precision-dynamics`](https://github.com/hamzasalmahi-lab/fnd-precision-dynamics),
[`fnd-ddm-model`](https://github.com/hamzasalmahi-lab/fnd-ddm-model),
[`fnd-active-inference`](https://github.com/hamzasalmahi-lab/fnd-active-inference)
— specialise the allostatic component to the seizure-firing regime and add
between-episode dynamics. The N3 sleep state in `HRIT_I_Intero_v4` is the
physiological reference for the FND models: where the FND patient's ICP
trajectory looks compared to where it should look during normal sleep.

## Running

Each script is standalone. From MATLAB:

```matlab
HRIT_I_Intero_v4        % the allostatic component, all six states
HRIT_I_Sim_v3            % hierarchical perceptual inference
HRIT_I_Id_v2             % self-modelling
HRIT_I_Prec_v2           % metacognitive monitoring
HRIT_Integration_v3      % CSV state space across the six states
```

Each prints a summary table to the command window and produces 1–3 figures.

## A note on what this is and isn't

This is a research implementation, not a finished tool. The architecture
is meant to be testable — every parameter has a clinical or empirical
interpretation, and every prediction is meant to be falsifiable. Where
the framework is wrong, the right response is to revise it, not to add
free parameters. Things that don't work as expected go into the README
and the next version, not into the appendix.
