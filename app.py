"""Web demo.

Tab 1: enter an applicant, then see what happens when only age or sex changes.
Tab 2: the audit results from results/.

    python app.py
"""
from pathlib import Path

import gradio as gr

from audit import data
from audit.models import build
from audit.scoring import typical_applicant, what_if

RESULTS = Path(__file__).resolve().parent / "results"

X, y, groups = data.load()
X_blind = X.drop(columns=data.PROTECTED_COLUMNS)
full = build("logreg", X).fit(X, y)
blind = build("logreg", X_blind).fit(X_blind, y)

base_row = typical_applicant(X)
status_by_sex = {
    sex: X.loc[groups["sex"] == sex, "personal_status"].mode()[0]
    for sex in ["female", "male"]
}


def check(age, sex, amount, months, checking, savings, employment, housing):
    overrides = {
        "credit_amount": amount,
        "duration": months,
        "checking_status": checking,
        "savings_status": savings,
        "employment": employment,
        "housing": housing,
    }
    return what_if(full, blind, base_row, status_by_sex, age, sex, overrides)


def options(col):
    return sorted(X[col].unique().tolist())


with gr.Blocks(title="Loan model fairness audit") as demo:
    gr.Markdown(
        "# Loan model fairness audit\n"
        "Prototype built on a public 1994 German credit dataset. Not financial advice."
    )
    with gr.Tab("Applicant what-if"):
        gr.Markdown(
            "The full model sees age and sex. The blind model has both columns removed. "
            "Rows below change only age or sex and leave everything else as entered."
        )
        with gr.Row():
            with gr.Column():
                age = gr.Slider(int(X["age"].min()), int(X["age"].max()), value=35, step=1, label="Age")
                sex = gr.Radio(["female", "male"], value="female", label="Sex")
                amount = gr.Number(value=float(X["credit_amount"].median()), label="Loan amount (DM)")
                months = gr.Slider(4, 72, value=int(X["duration"].median()), step=1, label="Duration (months)")
                checking = gr.Dropdown(options("checking_status"), value=base_row["checking_status"], label="Checking account")
                savings = gr.Dropdown(options("savings_status"), value=base_row["savings_status"], label="Savings")
                employment = gr.Dropdown(options("employment"), value=base_row["employment"], label="Time in current job")
                housing = gr.Dropdown(options("housing"), value=base_row["housing"], label="Housing")
                go = gr.Button("Score applicant")
            with gr.Column():
                table = gr.Dataframe(label="Probability the loan is repaid, and decision at a 0.5 cutoff")
        go.click(
            check,
            inputs=[age, sex, amount, months, checking, savings, employment, housing],
            outputs=table,
        )

    with gr.Tab("Audit results"):
        summary = RESULTS / "summary.md"
        gr.Markdown(
            summary.read_text() if summary.exists() else "Run `python -m audit.run_audit` first."
        )
        if (RESULTS / "gaps.png").exists():
            gr.Image(str(RESULTS / "gaps.png"), show_label=False)

if __name__ == "__main__":
    demo.launch()
