# Recorded GPU configuration

`baseline_t4.json` records the main 96-cubed, width-16 configuration. It is a reference record. The current CLI does not load this JSON automatically; the fresh-run notebook and documented command arguments supply the settings.

The effective loss was unweighted cross-entropy despite configured inverse-frequency weights in the trainer. The final short accumulation window also retains the archived four-step scaling. Read `../docs/KNOWN_LIMITATIONS.md` before changing or interpreting the configuration.

Earlier CPU verification used smaller inputs and a narrower model. Its results are deliberately kept outside this GPU-focused repository to avoid mixing configurations.
