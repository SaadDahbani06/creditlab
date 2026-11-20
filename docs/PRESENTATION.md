# Presentation material

## Two-minute walkthrough of the research site

*Setup: `streamlit run app/streamlit_app.py`; start on Overview. Every number below is read from
saved artifacts; check them on screen rather than quoting from memory.*

**0:00 – Framing (Overview).**
"CreditLab asks two things: how well six months of repayment history predicts next-month default
for 30,000 Taiwanese cardholders, and how much we should trust the portfolio tail-risk numbers
built on those predictions. Only the default probability is learned from data. The money figures
rest on stated assumptions and are not observed losses."

**0:20 – Protocol and models (Predictive Modeling).**
"I split clients 60/20/20, tuned by training cross-validation, and froze every choice on
validation before scoring the test split. One disclosed deviation: I had seen test metrics before
tightening the calibration rule. It didn't change the selected model or any test number, but I
don't call the test set untouched. XGBoost reaches AUC 0.790 against 0.780 for logistic regression.
The paired-bootstrap gap is +0.010, so it's small but real. Both models are already calibrated, so
recalibration wasn't adopted."

**0:55 – Tail risk (Portfolio Risk & Findings, top).**
"A one-factor Gaussian model turns PDs into a loss distribution. With independent defaults the
99% VaR is only 7% above expected loss. With a small correlation, ρ = 0.04, expected loss is
unchanged but VaR rises 52%, because bad months make many accounts default together."

**1:20 – Research extension (same page, lower half).**
"The usual calculation treats PDs as exact. I refitted the model on 200 bootstrap resamples of the
training data and ran every PD set through the same simulated draws. Individual PDs move a lot,
but that mostly cancels across 6,000 accounts. What remains is a portfolio-wide shift, driven
mostly by how many defaults land in the training sample. Under independence that uncertainty is as
large as the tail margin itself. Under correlation it is under 1% of loss variance, and the
correlation assumption, which one month of data can't estimate, matters roughly ten times more."

**1:50 – Limits (Methodology & Limitations).**
"The label is a missed payment, not a write-off, so the money figures are upper bounds. It is one
period, existing customers only, and there's no independent validation set. The extension is
post-hoc on the same data."
