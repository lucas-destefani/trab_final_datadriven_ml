# -*- coding: utf-8 -*-
"""Projeto Final - Rede neural para prever Cl, Cd e Cm de aerofólios."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

# Pasta com os dados, ao lado deste script
DB = Path(__file__).resolve().parent / "airfoil_db_v1_to_students"

airfoils = pd.read_csv(DB / "airfoils.csv")
polars   = pd.read_csv(DB / "polars.csv")

# União das tabelas pelo ID - mesmo número de linhas da tabela de polars
uni = pd.merge(polars, airfoils, on="id")

# Limpeza: remove linhas incompletas e Cd não positivo (log10 não existe)
uni = uni.dropna(subset=["Re", "alpha", "Cl", "Cd", "Cm"])
uni = uni[uni["Cd"] > 0]

# Log de Re e Cd
uni["LogRe"] = np.log10(uni["Re"])
uni["LogCd"] = np.log10(uni["Cd"])

# Filtros: separa o naca2412 (alvo) e limita a espessura
target = uni[uni["name"] == "naca2412"]
assert len(target) > 0, "naca2412 não encontrado em airfoils.csv"
uni = uni[uni["name"] != "naca2412"]
uni = uni[(uni["t_max"] >= 0.06) & (uni["t_max"] <= 0.16)]

# Split POR AEROFÓLIO (70% treino, 15% validação, 15% teste)
rng = np.random.default_rng(30)
ids = rng.permutation(uni["id"].unique())

n    = len(ids)
n_tr = int(n * 0.70)
n_v  = int(n * 0.15)

treino = uni[uni["id"].isin(ids[:n_tr])]
val    = uni[uni["id"].isin(ids[n_tr:n_tr + n_v])]
teste  = uni[uni["id"].isin(ids[n_tr + n_v:])]

X_columns = ['Au0', 'Au1', 'Au2', 'Au3', 'Au4', 'Au5',
             'Al0', 'Al1', 'Al2', 'Al3', 'Al4', 'Al5', 'alpha', 'LogRe']
Y_columns = ['Cl', 'LogCd', 'Cm']


# Matrizes X e Y de um DataFrame
def X_Y(df):
    X = df[X_columns].values.astype(np.float32)
    Y = df[Y_columns].values.astype(np.float32)
    return X, Y


# Padroniza com média e desvio do TREINO
def padroniza(X, Y):
    return (X - mx_tr) / sx_tr, (Y - my_tr) / sy_tr


X_tr, Y_tr = X_Y(treino)
mx_tr, sx_tr = X_tr.mean(axis=0), X_tr.std(axis=0)
my_tr, sy_tr = Y_tr.mean(axis=0), Y_tr.std(axis=0)
assert np.all(sx_tr > 0) and np.all(sy_tr > 0), "coluna constante no treino (desvio = 0)"

X_tr,     Y_tr     = padroniza(X_tr, Y_tr)
X_val,    Y_val    = padroniza(*X_Y(val))
X_teste,  Y_teste  = padroniza(*X_Y(teste))
X_target, Y_target = padroniza(*X_Y(target))

# Tensores
Entrada_treino, Saida_treino = torch.tensor(X_tr),    torch.tensor(Y_tr)
Entrada_val,    Saida_val    = torch.tensor(X_val),   torch.tensor(Y_val)
Entrada_teste                = torch.tensor(X_teste)
Entrada_target               = torch.tensor(X_target)

# Rede: 14 entradas -> 64 -> 64 -> 3 saídas
rede = nn.Sequential(
    nn.Linear(14, 64),   # camada 1: mistura as 14 entradas em 64 combinações
    nn.Tanh(),           # ativação: "dobra" o sinal, permitindo aprender curvas
    nn.Linear(64, 64),   # camada 2: combina as combinações
    nn.Tanh(),
    nn.Linear(64, 3)     # saída linear (sem tanh): Cl, LogCd, Cm podem ter qualquer valor
)

loss_fn    = nn.MSELoss()
otimizador = torch.optim.Adam(rede.parameters(), lr=1e-3)

# Minibatches
loader = DataLoader(TensorDataset(Entrada_treino, Saida_treino), batch_size=256, shuffle=True)

# Early stopping
melhor_val     = float("inf")   # recorde: qualquer erro bate
paciencia      = 20             # épocas sem melhorar toleradas
contador       = 0              # épocas seguidas sem melhorar
melhores_pesos = None

hist_treino = []
hist_val    = []

for epoca in range(500):
    rede.train()
    soma = 0
    for xb, yb in loader:
        loss = loss_fn(rede(xb), yb)
        otimizador.zero_grad()
        loss.backward()
        otimizador.step()
        soma += loss.item()

    rede.eval()
    with torch.no_grad():
        loss_val = loss_fn(rede(Entrada_val), Saida_val).item()

    hist_treino.append(soma / len(loader))
    hist_val.append(loss_val)

    if loss_val < melhor_val:                              # bateu o recorde?
        melhor_val = loss_val
        melhores_pesos = copy.deepcopy(rede.state_dict())  # guarda os pesos
        contador = 0
    else:
        contador += 1
        if contador >= paciencia:                          # esgotou a paciência
            print("Parou na época", epoca)
            break

rede.load_state_dict(melhores_pesos)
rede.eval()
print("Melhor erro de validação:", melhor_val)

# Curva de convergência
plt.plot(hist_treino, label="treino")
plt.plot(hist_val, label="validação")
plt.yscale("log")
plt.xlabel("época"); plt.ylabel("MSE"); plt.legend()
plt.show()

# ---------------- Métricas no conjunto de teste ----------------
with torch.no_grad():
    prev_teste = rede(Entrada_teste).numpy()      # previsão padronizada

prev_teste_real = prev_teste * sy_tr + my_tr      # previsão em unidades físicas
Y_teste_real    = Y_teste * sy_tr + my_tr         # respostas certas

rmse_cl = np.sqrt(np.mean((prev_teste_real[:, 0] - Y_teste_real[:, 0])**2))
rmse_cm = np.sqrt(np.mean((prev_teste_real[:, 2] - Y_teste_real[:, 2])**2))
cd_prev = 10 ** prev_teste_real[:, 1]             # volta do log
cd_real = 10 ** Y_teste_real[:, 1]
erro_cd = np.mean(np.abs(cd_prev - cd_real) / cd_real) * 100

print("RMSE cl:", rmse_cl)
print("RMSE cm:", rmse_cm)
print("Erro médio cd (%):", erro_cd)

# ---------------- NACA 2412 (nunca visto no treino) ----------------
with torch.no_grad():
    prev_target = rede(Entrada_target).numpy()

alpha_na = target["alpha"].values
Re_na    = target["Re"].values

prev_target_real = prev_target * sy_tr + my_tr
Y_target_real    = Y_target * sy_tr + my_tr

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

for Re_val in np.unique(Re_na):
    m = (Re_na == Re_val)
    ordem = np.argsort(alpha_na[m])

    a    = alpha_na[m][ordem]
    cl_x = Y_target_real[m, 0][ordem]           # XFOIL
    cl_r = prev_target_real[m, 0][ordem]        # rede
    cm_x = Y_target_real[m, 2][ordem]
    cm_r = prev_target_real[m, 2][ordem]
    cd_x = 10**Y_target_real[m, 1][ordem]
    cd_r = 10**prev_target_real[m, 1][ordem]

    # Quadro 0: curva de sustentação c_l(α)
    p = ax[0].plot(a, cl_x, "o", label=f"Re = {Re_val:.0e}")
    cor = p[0].get_color()                      # mesma cor para este Re
    ax[0].plot(a, cl_r, "-", color=cor)

    # Quadro 1: polar de arrasto
    ax[1].plot(cd_x, cl_x, "o", color=cor)
    ax[1].plot(cd_r, cl_r, "-", color=cor)

    # Quadro 2: momento c_m(α)
    ax[2].plot(a, cm_x, "o", color=cor)
    ax[2].plot(a, cm_r, "-", color=cor)

    # Região linear (|α| <= 4): inclinação c_lα e centro aerodinâmico
    lin = np.abs(a) <= 4
    if lin.sum() < 2:
        print(f"Re = {Re_val:.0e} | poucos pontos com |α| <= 4, pulando ajuste")
        continue

    cla_x = np.polyfit(a[lin], cl_x[lin], 1)[0] * 180 / np.pi   # por radiano
    cla_r = np.polyfit(a[lin], cl_r[lin], 1)[0] * 180 / np.pi
    xac_x = 0.25 - np.polyfit(cl_x[lin], cm_x[lin], 1)[0]       # x_ac = 0.25 - dCm/dCl
    xac_r = 0.25 - np.polyfit(cl_r[lin], cm_r[lin], 1)[0]

    print(f"Re = {Re_val:.0e} | c_lα XFOIL = {cla_x:.3f} | rede = {cla_r:.3f}")
    print(f"Re = {Re_val:.0e} | x_ac XFOIL = {xac_x:.4f} | rede = {xac_r:.4f}")

ax[0].set_xlabel("α [graus]"); ax[0].set_ylabel("c_l"); ax[0].set_title("Curva de sustentação")
ax[1].set_xlabel("c_d");       ax[1].set_ylabel("c_l"); ax[1].set_title("Polar de arrasto")
ax[2].set_xlabel("α [graus]"); ax[2].set_ylabel("c_m"); ax[2].set_title("Momento (1/4 de corda)")
ax[0].legend()

plt.tight_layout()
plt.show()
