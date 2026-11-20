# Data

## Source and license

*Default of Credit Card Clients*, UCI Machine Learning Repository, dataset ID 350.
Page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
DOI: 10.24432/C55S3H. Creator: I-Cheng Yeh.

License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
Required attribution:

> Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C55S3H

Introductory paper: Yeh, I.-C. & Lien, C.-H. (2009). The comparisons of data mining techniques
for the predictive accuracy of probability of default of credit card clients.
*Expert Systems with Applications*, 36(2), 2473–2480.

The data files are **not committed** to this repository (see `.gitignore`); they are downloaded
by `creditlab fetch`.

## Getting the data

```bash
creditlab fetch            # downloads via ucimlrepo into data/raw/uci_350_data.csv
```

If the programmatic download fails (no network, API change):

1. Download `data.csv` from https://archive.ics.uci.edu/static/public/350/data.csv
2. Either place it at `data/raw/data.csv` and run `creditlab fetch`, or run
   `creditlab fetch --local-file /path/to/data.csv`.

The file must have the official layout `ID, X1, …, X23, Y` (30,000 rows).

## Immutability

`data/raw/uci_350_data.csv` is written once and set read-only. Its sha256 and an
order-sensitive content hash are stored in `data/raw/uci_350_provenance.json`. Later runs
verify the hash and refuse to continue if the file has changed. The provenance also records
whether the content is identical to the official CSV (it is, for the current copy).
Overwriting requires `creditlab fetch --force`.

## Processed data

`creditlab prepare` writes `data/processed/` (semantic names, engineered features, the persisted
split and a DuckDB database). See `docs/DATA_DICTIONARY.md`.
