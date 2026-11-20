# Legacy interactive pages (not in the main site navigation)

These are the original CreditLab v1 dashboard pages, including the interactive
**Stress Testing** and **Policy Lab** tools. They were removed from the main research
site (decision D22) but are kept working because they exercise the same
library functions with user-chosen hypothetical inputs.

Run any page on its own from the project root, for example:

```bash
streamlit run app/legacy/5_Stress_Testing.py
streamlit run app/legacy/6_Policy_Lab.py
```

They are covered by `tests/test_app.py`.
