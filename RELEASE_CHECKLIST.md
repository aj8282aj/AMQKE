# AMQKE release checklist

- [ ] Confirm that the manuscript authors and relevant university/co-author/funder agreements permit the code to be released under MIT.
- [ ] Confirm the copyright holders named in `LICENSE`.
- [ ] Confirm the `v25`/“Version 26” source version label is intentional and synchronized across manuscript, notebook, and script.
- [ ] Supply the exact `heart.csv`, `diabetes.csv`, `banknote.csv`, and `sonar.csv` files used for the reported results in a separate, appropriately licensed location.
- [ ] Resolve the full-dataset imputation in `clean_numeric_features()` before making a strict leakage-free preprocessing claim; rerun all affected experiments after the correction.
- [ ] Confirm the UCI Heart dataset variant matches the manuscript's reported sample count.
- [ ] Decide whether to implement disk-based kernel caching and machine-readable CSV/JSON result exports, or revise the manuscript's reproducibility description to match the current code.
- [ ] Record exact Python and package versions used for the final run.
- [ ] Retain output figures and full experiment logs for the version associated with manuscript tables.
- [ ] Recheck Qiskit compatibility before upgrading beyond the pinned Qiskit version; feature-map classes used here are deprecated as of Qiskit 2.1.
- [ ] Add a DOI or publication details to `CITATION.cff` when available.
