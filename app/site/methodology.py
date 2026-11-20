"""Methodology & limitations: data, models, maths, validation, reproducibility, caveats."""

from __future__ import annotations

import streamlit as st

from creditlab.reporting import dashboard as d
from creditlab.utils.config import PROJECT_ROOT

cfg = d.config()
d.page_header(
    "Methodology & limitations",
    "How the results were produced, and what they cannot show",
    "A compact technical summary. The full derivations, decision log and independent review are in "
    "the project documents linked at the bottom of this page.")

st.header("Data", anchor=False)
st.markdown(
    "UCI dataset 350, *Default of Credit Card Clients* (Yeh, 2009; CC BY 4.0): 30,000 existing "
    "cardholders of one Taiwanese bank, with six months of history (April–September 2005) and an "
    "observed label for default on the October payment. The raw file is stored read-only and "
    "hash-verified. Undocumented status codes (−2 and 0, two-thirds of month-records) are kept as "
    "observed categories rather than reinterpreted. Sex, age, marital status and education are "
    "excluded from the models and used only for an aggregate audit.")

st.header("Models and validation", anchor=False)
st.markdown(
    "- **Split:** stratified 60/20/20 by client (18,000 / 6,000 / 6,000), fixed seed. All accounts "
    "share one observation window, so this measures generalisation to other clients of the same "
    "period, not to later periods.\n"
    "- **Models:** base-rate baseline; L2 logistic regression (one-hot status codes, signed-log "
    "amounts); XGBoost (randomized search over depth, learning rate, trees, regularisation). "
    "All preprocessing is fitted inside scikit-learn pipelines on training folds only.\n"
    "- **Selection:** 5-fold training cross-validation on log loss; calibration variant and "
    "primary model chosen on validation; test split scored afterwards with leakage checks "
    "(training-ID hash, split disjointness).\n"
    "- **Uncertainty in metrics:** 1,000-replicate bootstrap on the test split; paired bootstrap "
    "for the model difference.")

st.header("Portfolio loss model", anchor=False)
st.markdown("For account $i$ with probability of default $p_i$, assumed loss-given-default "
            "$\\mathrm{LGD}$ and exposure $\\mathrm{EAD}_i$, one-month expected loss is")
st.latex(r"\mathrm{EL}=\sum_i p_i\,\mathrm{LGD}\,\mathrm{EAD}_i ,\qquad "
         r"\mathrm{EAD}_i=B_i+\mathrm{CCF}\,(\mathrm{Limit}_i-B_i)^+ .")
st.markdown("Defaults are simulated with a one-factor Gaussian model. Account $i$ defaults when")
st.latex(r"\sqrt{\rho}\,F+\sqrt{1-\rho}\,\varepsilon_i<\Phi^{-1}(p_i),\qquad F,\varepsilon_i"
         r"\overset{iid}{\sim}N(0,1).")
st.markdown(
    "Every marginal PD is preserved for any $\\rho$, so correlation changes how defaults cluster "
    "but not expected loss. Conditional on $F$ defaults are independent; a bad draw of $F$ raises "
    "every account's PD at once, which is what fattens the tail. VaR is the empirical "
    "$\\alpha$-quantile of simulated losses and Expected Shortfall the tie-correct tail mean")
st.latex(r"\mathrm{ES}_\alpha=\mathrm{VaR}_\alpha+\frac{1}{n(1-\alpha)}\sum_{s=1}^{n}"
         r"\big(L_s-\mathrm{VaR}_\alpha\big)^+ .")

st.header("Research extension: PD estimation uncertainty", anchor=False)
st.markdown(
    "Fixed-PD risk numbers treat $\\hat p$ as exact. The extension refits the frozen model "
    "specification on $B$ bootstrap resamples of the training split, giving PD vectors "
    "$p^{(1)},\\dots,p^{(B)}$ for the same test portfolio, and simulates each one with the same "
    "random draws (common random numbers). Two summaries follow:")
st.latex(r"\text{spread of } \mathrm{VaR}_\alpha\big(p^{(b)}\big)\ \text{across } b"
         r"\qquad\text{and}\qquad F_{\text{mix}}(\ell)=\frac1B\sum_{b=1}^{B}"
         r"P\big(L\le \ell\,\big|\,p^{(b)}\big).")
st.markdown("The law of total variance separates the two sources of uncertainty:")
st.latex(r"\operatorname{Var}(L)=\underbrace{\mathbb{E}_b\!\left[\operatorname{Var}(L\mid p^{(b)})"
         r"\right]}_{\text{default randomness}}+\underbrace{\operatorname{Var}_b\!\left("
         r"\mathbb{E}[L\mid p^{(b)}]\right)}_{\text{PD estimation}} .")
st.markdown(
    "Bootstrapping clients captures sampling variability of the training data only. It does not "
    "address distribution shift to later periods, variables the dataset does not contain, the "
    "selection of who became a cardholder, or economic conditions absent from 2005. The bootstrap, "
    "Monte Carlo simulation and the Gaussian copula are established methods; the contribution here "
    "is an empirical investigation that applies them to this dataset.")

st.header("Validation status and reproducibility", anchor=False)
st.markdown(
    "- **Original evaluation (frozen).** All choices were made on validation before the test split "
    "was scored. One disclosed deviation: test metrics were displayed before the calibration rule "
    "was tightened (decision D9). The change did not alter the primary model or any test number. "
    "Original outputs are frozen with SHA-256 checksums in `results/baseline_v1/`.\n"
    "- **Post-hoc analyses.** The independent review and the uncertainty extension were carried "
    "out after the test results were known. They reuse the same data and are exploratory.\n"
    "- **No independent validation.** No new, untouched dataset exists for this project; a strong "
    "claim of generalisation would need one.\n"
    "- **Reproducibility.** One master seed with named sub-seeds; a full re-run reproduced every "
    "saved output byte for byte on the original machine.")
man_path = cfg.artifacts / "metrics" / "run_manifest.json"
if man_path.exists():
    man = d.read(cfg, "metrics/run_manifest.json")
    env = man["environment"]
    d.table(["Run provenance", ""], [
        ["Master seed", str(man["seed"])],
        ["Config fingerprint", man["config_fingerprint"]],
        ["Code version (git)", (env.get("git_commit") or "not recorded")[:12]],
        ["Python / scikit-learn / XGBoost", f"{env['python']} / {env['packages'].get('scikit-learn')}"
         f" / {env['packages'].get('xgboost')}"],
    ])
st.code("source .venv/bin/activate\n"
        "creditlab all --skip-fetch      # prepare → train → evaluate → simulate → policy → "
        "uncertainty → report\n"
        "pytest && ruff check .\n"
        "streamlit run app/streamlit_app.py", language="bash")

st.header("Limitations", anchor=False)
st.markdown(
    "- **Target ≠ loss event.** The label is a missed or late payment, not a charge-off. Applying "
    "a 60% LGD to every such event overstates losses heavily; money figures are best read as "
    "*loss if every missed payment became a 60% write-off*.\n"
    "- **Assumed, not estimated:** LGD, EAD, correlation ρ and stress multipliers. The tail is "
    "driven mainly by ρ, which one outcome month cannot identify.\n"
    "- **Single period and unknown sampling.** A 22% one-month default rate is far above any "
    "card portfolio; if defaults were over-sampled, the PDs are calibrated to the sample, not a "
    "population.\n"
    "- **Selection bias.** Existing, already-approved cardholders only; no applicants.\n"
    "- **Not causal, not a decision system.** Feature importance is association. The secondary "
    "policy comparison in the research report uses outcomes observed under the bank's actual "
    "treatment and is not evidence of profit or of a better lending rule.")

st.header("Documents", anchor=False)
docs = [("RESEARCH_REPORT.md", "Research report"), ("RESEARCH_EXTENSION.md", "Research extension"),
        ("INDEPENDENT_REVIEW.md", "Independent review"),
        ("MODEL_CARD.md", "Model card"), ("DECISIONS.md", "Decision log")]
cols = st.columns(len(docs))
for col, (fname, label) in zip(cols, docs, strict=True):
    path = PROJECT_ROOT / "docs" / fname
    if path.exists():
        col.download_button(label, path.read_bytes(), file_name=fname, mime="text/markdown",
                            width="stretch")
repo = cfg.raw.get("links", {}).get("repository")
st.caption(f"Source code: {repo}" if repo else
           "Source code: repository link not yet published. The project runs locally from the "
           "command line; see the README.")
d.footer(cfg)
