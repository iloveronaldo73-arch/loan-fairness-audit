"""Synthetic stand-in with the same columns as credit-g. Only for testing the
code paths; nothing in results/ or the writeup comes from this."""
import numpy as np
import pandas as pd


def make_german(n=800, seed=0):
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


def make_hmda(n=6000, seed=0):
    """HMDA-shaped frame: right column names and messy values, made-up numbers."""
    rng = np.random.default_rng(seed)
    race = rng.choice(["White", "Black or African American", "Asian", "Race Not Available"], n, p=[0.6, 0.12, 0.1, 0.18])
    sex = rng.choice(["Male", "Female", "Joint", "Sex Not Available"], n, p=[0.4, 0.3, 0.25, 0.05])
    age = rng.choice(["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74", "8888"], n)
    income = rng.gamma(3, 30, n).round()
    amount = (income * rng.uniform(1.5, 4, n) * 1000).round(-3)
    ltv = rng.uniform(40, 100, n).round(1)
    logit = 2.0 + 0.01 * (income - 90) - 0.03 * (ltv - 80) - 0.5 * (race == "Black or African American")
    approved = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "action_taken": np.where(approved, 1, 3),
            "loan_purpose": 1, "occupancy_type": 1, "lien_status": 1, "reverse_mortgage": 2,
            "open-end_line_of_credit": 2, "business_or_commercial_purpose": 2,
            "loan_amount": amount, "income": income, "property_value": (amount / 0.8).round(-3),
            "combined_loan_to_value_ratio": ltv, "loan_term": 360,
            "tract_to_msa_income_percentage": rng.integers(40, 200, n),
            "debt_to_income_ratio": rng.choice(["<20%", "20%-<30%", "30%-<36%", "40", "50%-60%", ">60%"], n),
            "loan_type": rng.choice([1, 2, 3], n), "derived_loan_product_type": "Conventional:First Lien",
            "conforming_loan_limit": "C", "preapproval": rng.choice([1, 2], n),
            "derived_race": race, "derived_ethnicity": "Not Hispanic or Latino", "derived_sex": sex,
            "applicant_age": age,
            "interest_rate": np.where(approved, 6.5, np.nan),  # only filled for approved loans: must not be used
            "denial_reason-1": np.where(approved, "", "3"),
        }
    )
    df.loc[rng.random(n) < 0.05, "income"] = np.nan
    df["income"] = df["income"].astype(object)
    df.loc[rng.random(n) < 0.02, "income"] = "Exempt"
    return df
