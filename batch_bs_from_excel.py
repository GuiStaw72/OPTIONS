

# -*- coding: utf-8 -*-
"""
Script batch pour calculer les prix Black-Scholes ET les Greeks (call/put)
à partir d'un fichier Excel.

Usage:
    python batch_bs_from_excel.py Input_BS.xlsx Output_BS_with_Greeks.xlsx

Colonnes attendues:
    TICKER, S, K, T, r, q, sigma
"""

import sys
import pandas as pd
import numpy as np
from Calc_GreeksPrIV import (
    bs_price, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho
)

def _to_decimal(x):
    try:
        val = float(x)
    except Exception:
        return np.nan
    return val/100.0 if val > 1.0 else val

def run(in_file: str, out_file: str) -> None:
    df = pd.read_excel(in_file, engine="openpyxl")
    if "r" not in df.columns or "q" not in df.columns:
        raise ValueError("Le fichier doit contenir les colonnes 'r' et 'q'.")

    df["r_dec"], df["q_dec"] = df["r"].apply(_to_decimal), df["q"].apply(_to_decimal)

    cols = [
        "Call_Price","Delta_Call","Theta_Call","Rho_Call",
        "Put_Price","Delta_Put","Theta_Put","Rho_Put",
        "Gamma","Vega"
       
    ]

    def _compute(row):
        try:
            S = float(row["S"]); K = float(row["K"]); T = float(row["T"])
            r = float(row["r_dec"]); q = float(row["q_dec"]); sigma = float(row["sigma"])
            call = bs_price(S, K, T, r, q, sigma, "call")
            put  = bs_price(S, K, T, r, q, sigma, "put")
            d_call = bs_delta(S, K, T, r, q, sigma, "call")
            d_put  = bs_delta(S, K, T, r, q, sigma, "put")
            gamma  = bs_gamma(S, K, T, r, q, sigma)
            vega   = bs_vega(S, K, T, r, q, sigma)
            t_call = bs_theta(S, K, T, r, q, sigma, "call")
            t_put  = bs_theta(S, K, T, r, q, sigma, "put")
            r_call = bs_rho(S, K, T, r, q, sigma, "call")
            r_put  = bs_rho(S, K, T, r, q, sigma, "put")
            return pd.Series({
                "Call_Price": call, "Delta_Call": d_call,"Theta_Call": t_call,"Rho_Call": r_call,
                "Put_Price": put,"Delta_Put": d_put,"Theta_Put": t_put,"Rho_Put": r_put,
                "Gamma": gamma, "Vega": vega
            })
        except Exception:
            return pd.Series({c: np.nan for c in cols})

    res = df.apply(_compute, axis=1)
    out_df = pd.concat([df.drop(columns=["r_dec","q_dec"]), res], axis=1)
    out_df.to_excel(out_file, index=False, engine="openpyxl")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_bs_from_excel.py Input_BS.xlsx Output_BS_with_Greeks.xlsx")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
