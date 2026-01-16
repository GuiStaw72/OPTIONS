
import pandas as pd

# Suppose que tu as déjà copié les fonctions Black-Scholes dans black_scholes.py
from Calc_GreeksPrIV import bs_price

# Charger les données
df = pd.read_csv("options_100_us.csv")

# Calculer le prix théorique Black-Scholes pour chaque option
df["bs_price"] = df.apply(
    lambda row: bs_price(
        S=row["S"],
        K=row["K"],
        T=row["T"],
        r=row["r"],
        q=row["q"],
        sigma=row["sigma"],
        opt_type=row["type"]
    ),
    axis=1
)

# Calculer l'écart avec le prix de marché
df["diff"] = df["bs_price"] - df["market_price"]

# Afficher les 10 plus gros écarts
print(df.sort_values("diff", key=abs, ascending=False).head(10))
