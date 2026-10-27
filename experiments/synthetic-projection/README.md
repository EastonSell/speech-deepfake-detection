# Synthetic projection diagnostic (exploratory)

This self-contained diagnostic explores a projection-rank control relevant to a possible future speech study. It uses generated Gaussian vectors only: **no audio, speech detector, dataset, or checkpoint was evaluated**, and the results do not establish real-speech accuracy. The follow-on was designed after observing the original synthetic result; local protocol freezing is not independent preregistration. Random row-space controls match PCA's training rank but not removed variance, retained spectrum, or conditioning. Ten projections share one dataset and are not ten independent replications. See the protocols and interpretation for limitations and a proposed real-speech evaluation plan, not an executed study.

## Reproduce

Use normal Python (not `python -O`): guards use `assert` and optimization disables them. The recorded run used Python 3.12.14, NumPy 2.3.5 and OpenBLAS 0.3.30. With NumPy installed:

```sh
python -m unittest -v test_rank_matched
python diagnostic.py > /tmp/diagnostic-results.json
python rank_matched.py > /tmp/rank-matched-results.json
```

Compare with the checked-in JSON files. Exact floating-point values or seeded controls can differ across SVD/BLAS implementations; inspect ranks and tolerances rather than assuming bitwise portability. Independent reviewers checked the synthetic setup and follow-on; the latter prompted stricter absolute projector checks and a deletion-rank guard. Three unit tests pass in the recorded environment. No data or checkpoint execution is required.
