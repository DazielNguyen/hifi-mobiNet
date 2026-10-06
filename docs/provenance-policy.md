# Provenance policy

Each imported file records a workspace-relative source locator, source Git HEAD
when available, working-tree state, SHA-256 and destination. Source HEAD identifies
the inspected repository, not necessarily the commit that trained a checkpoint.
Local machine paths belong in ignored `.local/` records only.

Evidence classes:

- `current_source`: inspected implementation at import time.
- `historical_source_candidate`: recovered or archived source without complete
  run-time binding.
- `run_bound_artifact`: an artifact identified by a contemporaneous or explicitly
  dated engineering record; the scope of that binding must be stated.
- `historical_result`: original stored observations, preserved byte for byte.
- `prior_audit`: a previously derived analysis, not a new experiment.
- `new_validation`: functional checks of this repository, never paper benchmarks.

Unchanged imports are committed before adaptation. `vendor/` is reference source,
not an executable promise; adapted modules are documented separately. Historical
logs that contain host paths remain outside Git, with source locators and hashes.
No scientific claim in an imported comment is automatically endorsed.

Result and artifact identities are independent: a valid checkpoint hash or a new
inference smoke test does not retroactively bind an old score to new code.
