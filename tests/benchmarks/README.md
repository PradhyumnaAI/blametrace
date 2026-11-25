# Planted-shortcut benchmarks

Each milestone ships with one benchmark here. A milestone is complete only when
its benchmark passes in CI. Benchmarks must be marked `@pytest.mark.benchmark`,
seeded, deterministic across five or more seeds, and CPU-runnable.

| Milestone | File | Planted shortcut |
|---|---|---|
| M1 | `test_m1_ann_shortcut.py` | leaked tabular column |
| M1.x | `test_m1x_slice_attribution.py` | leak limited to one metadata group |
| M2 | `test_m2_cnn_corner_shortcut.py` | corner watermark or glare |
| M3 | `test_m3_rnn_prefix_trigger.py` | boilerplate prefix |
| M4 | `test_m4_transformer_sink_patch.py` | attention sink or corner patch |
| M5 | `test_m5_cnn_mamba_border.py` | segmentation border artifact |

Every benchmark also includes a negative control: a model trained without the
shortcut must yield no significant culprits.

See `FREEZE.md` section 6 for the exact gate conditions.
