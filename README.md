# trab_final_datadriven_ml

Surrogate e POD-NN para a aerodinâmica de aerofólios. Projeto final de Data-Driven Modeling and Machine Learning for Engineering (PUCPR), prof. Gustavo L. O. Halila.

## Contexto

Para projetar uma asa é preciso conhecer as forças na seção transversal, o aerofólio, em muitas formas e condições de voo. Um solver de escoamento leva de segundos a horas por caso. O projeto treina modelos que dão a mesma informação em milissegundos, a partir de uma base com cerca de 225 mil resultados do XFOIL. Todos usam o mesmo aerofólio-alvo, o NACA 2412, que fica fora do treino, da validação e do teste e só aparece no final como caso nunca visto.

Os modelos trabalham com coeficientes aerodinâmicos adimensionais, por unidade de envergadura: sustentação (cl), arrasto (cd) e momento de arfagem em torno de 1/4 da corda (cm). O número de Reynolds afeta sobretudo o arrasto, e a base cobre Re = 0,5; 1; 2 e 4 × 10⁶. O coeficiente de pressão Cp descreve a pressão ponto a ponto na superfície, e a curva Cp(x/c) inteira é o alvo do estágio 2.

## Os dois estágios

### Estágio 1: surrogate

Uma rede neural recebe 14 entradas (12 coeficientes CST da forma, o ângulo de ataque alpha e log10 Re) e devolve 3 saídas (cl, log10 cd e cm). As camadas têm 14, 64, 64 e 3 neurônios, com tanh nas ocultas. O treino usa MSE, Adam, mini-lotes de 256 e parada antecipada na validação. O cd entra em log10 porque o MSE só enxerga a diferença absoluta, e o que importa é o erro relativo (um erro de 0,001 pesa 10 vezes mais em cd = 0,005 do que em cd = 0,05).

A avaliação usa o RMSE de cl e cm e o erro relativo médio de cd nos aerofólios de teste. Para a NACA 2412, o código compara a rede com o XFOIL em cl(alpha), na polar de arrasto e em cm(alpha), e calcula a inclinação da curva de sustentação (cl_alpha) e o centro aerodinâmico na faixa |alpha| ≤ 4°.

### Estágio 2: POD-NN

A curva de Cp tem 200 valores, mas as curvas que ocorrem são combinações de poucos padrões. A SVD das curvas de treino, depois de subtrair a curva média, dá os modos POD, e 17 modos guardam 99,9% da energia. A projeção de cada curva nos modos dá 17 coeficientes, que servem de gabarito. Uma rede com 13 entradas (12 CST e alpha) e 17 saídas aprende a prever esses coeficientes, e a curva volta a ser montada somando os modos à média. O modelo só vale para Re = 10⁶ e 0° ≤ alpha ≤ 8°, porque os dados de Cp só existem nessa faixa.

Os requisitos adicionais para pós-graduação (seção 7 do enunciado) não se aplicam a este trabalho.

## A base de dados

A pasta `airfoil_db_v1_to_students` vem da disciplina.

| Arquivo | Conteúdo |
| --- | --- |
| `airfoils.csv` | 3.455 aerofólios, um por linha: id, nome, origem, espessura e curvatura máximas e os 12 coeficientes CST |
| `polars.csv` | 225.283 resultados do XFOIL: id, Re, alpha, Cl, Cd e Cm (em torno de 1/4 da corda) |
| `cp_Re1e6.npz` | 16.789 curvas de Cp de 200 pontos (100 na superfície superior, 100 na inferior), só para Re = 10⁶ e alpha = 0, 2, 4, 6 e 8° |
| `airfoil_db.py`, `README.md`, `cleaning_log.txt` | funções auxiliares, descrição da base e lista das polares removidas na limpeza |

Os aerofólios são 1.955 reais, do UIUC Airfoil Coordinates Database, e 1.500 sintéticos, feitos por pequenas perturbações aleatórias dos reais. O XFOIL rodou em Re = 0,5; 1; 2 e 4 × 10⁶ e alpha de −4° a 12° em passos de 1°. A camada limite foi forçada a ser turbulenta a partir de 5% da corda, o que equivale a uma asa com o bordo de ataque levemente rugoso e evita saltos numéricos. Na limpeza saíram os pontos que não convergiram e 122 polares com arrasto fisicamente impossível.

Cada forma vira 12 números pela parametrização CST: cada superfície é y/c = √x (1 − x) · Σ Aₖ Bₖ(x), com k de 0 a 5 e Bₖ polinômios de Bernstein. Os coeficientes Au0 a Au5 (superior) e Al0 a Al5 (inferior) descrevem a forma exata que o XFOIL recebeu.

Regras de uso:

1. Só entram aerofólios com 0,06 ≤ t/c ≤ 0,16, porque os mais grossos mostram separação exagerada no XFOIL com transição forçada.
2. A NACA 2412 fica fora do treino, da validação e do teste.
3. O split é por aerofólio (70% treino, 15% validação, 15% teste), e todos os pontos de um aerofólio vão para o mesmo conjunto.

Depois do filtro sobram 3.102 aerofólios: 2.171 no treino, 465 na validação e 466 no teste. No estágio 2, 2.135, 459 e 459 deles têm curvas de Cp.

## Resultados

| Estágio 1 (teste, 466 aerofólios) | Valor |
| --- | --- |
| RMSE de cl | 0,013 |
| RMSE de cm | 0,002 |
| Erro relativo médio de cd | 1,3% |

Na NACA 2412 o RMSE de cl é 0,004 e o erro de cd é 0,5%. A inclinação cl_alpha fica entre 5,80 e 6,12 por radiano, com erro menor que 0,4% contra o XFOIL, e o centro aerodinâmico fica perto de 0,23 da corda, com diferença de até 0,002.

| Estágio 2 | RMSE de Cp |
| --- | --- |
| Piso de projeção (17 modos, teste) | 0,015 |
| POD-NN, aerofólios de teste | 0,026 |
| POD-NN, NACA 2412 (piso: 0,006) | 0,012 |

O piso é o erro de reconstruir as curvas de teste com os coeficientes exatos, o melhor que uma rede perfeita faria com 17 modos.

## Estrutura da pasta

```
tde/
├── README.md
├── requirements.txt
├── airfoil_db_v1_to_students/   base de dados
├── codigo/
│   ├── projeto_final.py
│   └── projeto_final.ipynb
├── apresentacao/
│   ├── Apresentacao_Projeto_Final_PT.pptx
│   └── Final_Project_Presentation_EN.pptx
├── figuras/                     figuras geradas pelo código
└── old/                         versões antigas
```

## Como rodar

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd codigo
python projeto_final.py
```

O código procura a pasta `airfoil_db_v1_to_students` ao lado de `codigo` ou uma pasta acima, e grava as figuras em `figuras`. Dá para trocar os caminhos com as variáveis de ambiente `DB` e `FIG`. O notebook `projeto_final.ipynb` tem o mesmo código. No Colab, o código monta o Drive e lê os arquivos de `MyDrive/Data Driven`. O treino do estágio 1 leva alguns minutos em CPU.
