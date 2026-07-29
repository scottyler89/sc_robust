# Decision 001: PyDESeq2 Sparse-Fit Safeguards

Status: implementation reviewed and vendored in sc_robust `0.2.0`.

## Evidence

- Clean base: `5196ce1` in the local PyDESeq2 repository.
- Reviewed implementation commit: `8d1c39f7bf17aabe105458b02040f8c9770f8637`.
- The pre-existing IRLS safeguard is the committed `1e-6` diagonal ridge at
  `780b48ec`; it is preserved and emits `irls_ridge_applied` diagnostics.
- The formalized Cox-Reid safeguard retries one singular Fisher-information
  inversion with `1e-6 * I` and emits `cox_reid_ridge_retry` diagnostics.
- The reviewed source is distributed in-repository and is not resolved from a
  sibling checkout at runtime.

## Numerical Policy

The safeguards are denominator/inversion protection for poor fits, not a new
regularized estimator. Ordinary well-conditioned utility outputs were compared
byte-for-byte against the stock clean checkout. Sparse fallback, singular retry,
and full historical regression evidence are recorded in the production-readiness
plan.

## Release Gate

The reviewed PyDESeq2 source is vendored under `pydeseq2/`, distributed with its
MIT license, and identified by `__source_commit__`. It is not listed as an
external dependency because the package ships the reviewed implementation.
Wheel, import, source-identity, and license checks are part of release CI.


## Ordinary-Fit Comparison
Using the same ordered synthetic fixture (`counts=[3,4,5,6]`, unit size factors, intercept plus two-level condition design), stock `5196ce1c` and hardened `8d1c39f` produced byte-identical serialized beta, mu, H, alpha, and convergence output.
```json
{"H":[0.4999997571429704,0.4999997571429704,0.499999904545492,0.499999904545492],"alpha":1.761851670369137e-06,"alpha_converged":true,"beta":[1.2527627740208185,0.4519852319295172],"irls_converged":true,"mu":[3.499999319339143,3.499999319339143,5.499999525415528,5.499999525415528]}
```
