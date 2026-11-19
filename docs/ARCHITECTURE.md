# Architecture

Core owns data identities, local conversion, simulation, scoring reference FK, scheduling, metrics, traces and independent verification.

Model repositories own weight provenance, weight conversion, input construction, reference features, recurrent state, inference and PD targets. They depend on a pinned Core submodule and do not import sibling model repositories.

The one-command entry point resolves and verifies original artifacts, validates pinned assets, builds or reuses converted weights, prepares the selected dataset, runs batch simulation and independently verifies the result. Locks and atomic receipts coordinate simultaneous evaluations.

Results preserve runtime/protocol snapshots and source, model, dataset and trace hashes. Reproduction compares complete verified runs with matching execution identities and records numerical differences.
