

import math
from typing import Literal, Tuple, Optional

OptionType = Literal["call", "put"]

def _norm_cdf(x: float) -> float:
    """CDF de la loi normale standard via erf (évite dépendances externes)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    """PDF de la loi normale standard."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    """Calcule d1 et d2 en Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T, sigma doivent être > 0")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, opt_type: OptionType = "call") -> float:
    """Prix Black-Scholes pour option européenne (call/put) avec dividende continu q."""
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    if opt_type == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    elif opt_type == "put":
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)
    else:
        raise ValueError("opt_type doit être 'call' ou 'put'.")

def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, opt_type: OptionType = "call") -> float:
    """Delta (sensibilité au spot)."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    if opt_type == "call":
        return math.exp(-q * T) * _norm_cdf(d1)
    else:
        return math.exp(-q * T) * (_norm_cdf(d1) - 1.0)

def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Gamma (sensibilité du delta)."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * math.sqrt(T))

def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Vega (sensibilité à la volatilité), par point (1.0 = 100% de vol)."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)

def bs_theta(S: float, K: float, T: float, r: float, q: float, sigma: float, opt_type: OptionType = "call") -> float:
    """Theta (sensibilité au temps, par an)."""
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    first = -(S * math.exp(-q * T) * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if opt_type == "call":
        second = q * S * math.exp(-q * T) * _norm_cdf(d1)
        third = r * K * math.exp(-r * T) * _norm_cdf(d2)
        return first - second - third
    else:
        second = q * S * math.exp(-q * T) * _norm_cdf(-d1)
        third = r * K * math.exp(-r * T) * _norm_cdf(-d2)
        return first + second - third

def bs_rho(S: float, K: float, T: float, r: float, q: float, sigma: float, opt_type: OptionType = "call") -> float:
    """Rho (sensibilité au taux d'intérêt)."""
    _, d2 = d1_d2(S, K, T, r, q, sigma)
    if opt_type == "call":
        return K * T * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return -K * T * math.exp(-r * T) * _norm_cdf(-d2)

def implied_vol_newton(
    target_price: float,
    S: float, K: float, T: float, r: float, q: float,
    opt_type: OptionType = "call",
    init_sigma: float = 0.2,
    max_iter: int = 100,
    tol: float = 1e-8
) -> Optional[float]:
    """
    Volatilité implicite via Newton-Raphson (avec repli binaire si Newton diverge).
    Retourne None si échec.
    """
    sigma = max(init_sigma, 1e-6)
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, q, sigma, opt_type)
        vega = bs_vega(S, K, T, r, q, sigma)
        if vega < 1e-12:  # éviter division par ~0
            break
        diff = price - target_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        # garder sigma raisonnable
        sigma = max(sigma, 1e-6)
        if sigma > 5.0:  # 500% de vol: peu probable
            sigma = 5.0

    # Repli : recherche binaire sur [low, high]
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price = bs_price(S, K, T, r, q, mid, opt_type)
        if abs(price - target_price) < tol:
            return mid
        if price > target_price:
            high = mid
        else:
            low = mid
    return None

# Petit démonstrateur
if __name__ == "__main__":
    S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.02, 0.01, 0.25
    call = bs_price(S, K, T, r, q, sigma, "call")
    put  = bs_price(S, K, T, r, q, sigma, "put")
    print(f"Call: {call:.4f}, Put: {put:.4f}")

    delta_c = bs_delta(S, K, T, r, q, sigma, "call")
    gamma   = bs_gamma(S, K, T, r, q, sigma)
    vega    = bs_vega(S, K, T, r, q, sigma)
    theta_c = bs_theta(S, K, T, r, q, sigma, "call")
    rho_c   = bs_rho(S, K, T, r, q, sigma, "call")
    print(f"Delta(call): {delta_c:.4f}, Gamma: {gamma:.6f}, Vega: {vega:.4f}, Theta(call): {theta_c:.4f}, Rho(call): {rho_c:.4f}")

    # Exemple de vol implicite depuis un prix marché
    target_call_price = call  # suppose qu'on observe ce prix sur le marché
    iv = implied_vol_newton(target_call_price, S, K, T, r, q, "call")
    print(f"Vol implicite (call): {iv:.4f}")

