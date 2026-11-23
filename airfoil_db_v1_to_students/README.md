# Airfoil database for the final project (v1)

Data-Driven Modeling and Machine Learning for Engineering, PUCPR.

The project statement describes the tasks. This file describes the data.

## Contents

| File | Description |
|---|---|
| `airfoils.csv` | One row per airfoil: `id`, `name`, `source` (`uiuc` or `synthetic`), `parent` (UIUC airfoil a synthetic one was perturbed from), geometry descriptors (`t_max`, `x_tmax`, `camber_max`, `x_cmax`), `cst_rms` (fit error of UIUC coordinates, chord units), and the 12 CST coefficients `Au0..Au5`, `Al0..Al5` |
| `polars.csv` | One row per converged XFOIL point: `id`, `Re`, `alpha` (deg), `Cl`, `Cd`, `Cm` (quarter chord) |
| `cp_Re1e6.npz` | Pressure distributions at Re = 1e6, alpha = 0, 2, 4, 6, 8 deg: arrays `id`, `alpha`, `x` (100-point cosine grid) and `cp` (columns 0-99 upper surface, 100-199 lower surface, both from LE to TE). NumPy archive: read it with `np.load`, do not unzip it |
| `airfoil_db.py` | Helpers: load the tables, rebuild coordinates from CST coefficients, load Cp |
| `cleaning_log.txt` | List of the polars removed in v1 (see below) |

The files are linked by the column `id` (one number per airfoil).

## How the data were generated

- Geometry: 1955 airfoils from the UIUC coordinate database (6% <= t/c <= 20%, CST fit RMS <= 2e-3 c), plus 1500 synthetic airfoils obtained by Gaussian perturbation (0.15 standard deviation per coefficient) of UIUC CST vectors, subject to the same thickness limits.
- Parameterization: class-shape transformation (CST), class function sqrt(x)(1 - x), 6 Bernstein coefficients per surface, closed trailing edge. XFOIL was run on the CST-reconstructed geometry, so the CST vector is an exact description of the analyzed shape.
- Solver: XFOIL 6.99, 160 panels, M = 0, transition forced at x/c = 0.05 on both surfaces, Ncrit = 9.
- Conditions: Re = 0.5, 1, 2 and 4 x 1e6; alpha from -4 to 12 deg in 1 deg steps.
- Non-converged points were removed (96.7% of the attempted points converged).
- v1 cleaning: XFOIL occasionally converges to a spurious solution in which the forced transition is not effective, giving drag far below the fully turbulent skin-friction level. Every polar (airfoil, Re) with at least one point below 0.9 x 2Cf, where 2Cf = 2 x 0.074 Re^-0.2 is the turbulent flat-plate skin friction of both surfaces, was removed: 122 polars, 0.8% of the points. The Cp snapshots of the removed Re = 1e6 polars were removed as well.

## Using the helpers

```python
import airfoil_db as adb

# Airfoil table and polar table
af, polars = adb.load()

# Coordinates of one airfoil, rebuilt from its 12 CST coefficients
x, y = adb.coords(af.iloc[0])

# Cp curves: airfoil id, angle of attack, x/c grid and Cp (one row per curve, 200 columns)
cp_id, cp_alpha, x_grid, cp = adb.load_cp()
```

## Known limitations

- Fully turbulent boundary layers (forced transition): no laminar drag bucket. Drag levels are representative of a rough or contaminated leading edge.
- Thick airfoils (t/c above about 0.16) and airfoils with maximum thickness aft of 45% chord show trailing-edge separation already at low alpha, with reduced lift-curve slope.
- XFOIL is a viscous-inviscid interaction code; results near stall are less reliable than in the linear range.
- Cp data exist only at Re = 1e6 and alpha = 0, 2, 4, 6, 8 deg: a model trained on them has no Reynolds-number input and is valid only there.

## Rules for using the data

1. **Thickness envelope:** use only airfoils with 0.06 <= t/c <= 0.16 (column `t_max`).
2. **Target airfoil:** keep the NACA 2412 (`name == "naca2412"`) out of training, validation and test. It is used only at the end, as a completely unseen case.
3. **Split by airfoil, never by point:** split the training, validation and test sets **by airfoil (`id`)**, never by individual polar points. Points from the same airfoil in different sets make the test error look much better than it is.
