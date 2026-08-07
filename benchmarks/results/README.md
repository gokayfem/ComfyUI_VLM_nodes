# Result artifacts

Committed JSON files are immutable raw benchmark evidence. Each artifact
contains environment identity, input dimensions and token counts, warmups,
every measured sample, full model output, quality-gate details, percentiles,
VRAM, and speedups.

Console logs and scratch experiments are ignored. Promote a result by rerunning
the benchmark with its final runner and committing the resulting JSON rather
than editing an artifact by hand.
