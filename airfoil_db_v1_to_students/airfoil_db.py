"""Helper functions for the final-project airfoil database (beta).

Usage
-----
    import airfoil_db as adb
    af, polars = adb.load()
    x, y = adb.coords(af.loc[0])
"""
import numpy as np
import pandas as pd
from math import comb

# Number of CST (Bernstein) coefficients per surface
N_CST = 6


def load(folder="."):
    # Airfoil table (geometry) and polar table (aerodynamics), joined later by "id"
    af = pd.read_csv(f"{folder}/airfoils.csv", keep_default_na=False)
    polars = pd.read_csv(f"{folder}/polars.csv")
    return af, polars


def cst_surface(A, x):
    # Class function sqrt(x)(1-x) times Bernstein shape function; closed trailing edge
    B = np.stack([comb(N_CST - 1, k) * x**k * (1 - x)**(N_CST - 1 - k) for k in range(N_CST)], axis=1)
    return np.sqrt(x) * (1 - x) * (B @ np.asarray(A, float))


def coords(row, n=121):
    # Rebuild Selig-ordered coordinates (TE -> upper -> LE -> lower -> TE) from one table row
    Au = [row[f"Au{i}"] for i in range(N_CST)]
    Al = [row[f"Al{i}"] for i in range(N_CST)]
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yu, yl = cst_surface(Au, x), cst_surface(Al, x)
    return np.r_[x[::-1], x[1:]], np.r_[yu[::-1], yl[1:]]


def load_cp(folder="."):
    # Cp on a fixed cosine grid: columns 0-99 upper surface, 100-199 lower surface (LE -> TE)
    d = np.load(f"{folder}/cp_Re1e6.npz")
    return d["id"], d["alpha"], d["x"], d["cp"]
