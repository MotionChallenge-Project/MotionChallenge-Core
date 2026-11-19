# Batch evaluation

Default world count is min(number_of_motions, 4096). Motions are scheduled longest first and finished environments are refilled. Policies execute the full batch; histories and recurrent state are indexed by environment.

The simulator performs four steps per control period. Scoring reference body poses are computed once before rollout and cached. GPU scoring accumulates full-duration and prefix metrics independently per motion. CPU verification checks recorded traces.

Report conversion, reference preparation, simulation and verification times separately. Compare throughput using active control transitions, environment count, dataset, checkpoint, simulator settings and trace mode. Concurrent GPU jobs are not an isolated performance measurement.

Use a local SSD cache when shared storage has slow synchronous writes. Converted SEED poses occupy about 16.4 GB and the scoring reference cache about 14.8 GB.
