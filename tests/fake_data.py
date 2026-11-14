"""Synthetic stand-in with the same columns as credit-g. Only for testing the
code paths; nothing in results/ or the writeup comes from this."""
import numpy as np
import pandas as pd


def make(n=800, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(19, 70, n)
    female = rng.random(n) < 0.3
    status = np.where(
        female, "female div/dep/mar", rng.choice(["male single", "male mar/wid", "male div/sep"], n)
    )
    young = age < 25
    housing = np.where(
        young,
        rng.choice(["for free", "rent", "own"], n, p=[0.4, 0.5, 0.1]),
        rng.choice(["for free", "rent", "own"], n, p=[0.05, 0.2, 0.75]),
    )
    amount = rng.gamma(2.0, 1500, n).round()
    duration = rng.integers(6, 48, n)
    checking = rng.choice(["<0", "0<=X<200", ">=200", "no checking"], n)
    savings = rng.choice(["<100", "100<=X<500", ">=1000", "no known savings"], n)
    employment = rng.choice(["<1", "1<=X<4", ">=7", "unemployed"], n)

    logit = (
        0.9
        + 0.8 * (housing == "own")
        - 0.0003 * amount
        - 0.02 * duration
        + 0.01 * (age - 35)
        + 0.5 * (checking == "no checking")
    )
    good = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "checking_status": checking,
            "duration": duration,
            "credit_amount": amount,
            "savings_status": savings,
            "employment": employment,
            "personal_status": status,
            "age": age,
            "housing": housing,
            "foreign_worker": rng.choice(["yes", "no"], n, p=[0.95, 0.05]),
            "class": np.where(good, "good", "bad"),
        }
    )
