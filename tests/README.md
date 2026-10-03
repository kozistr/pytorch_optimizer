# Test layout

`fixtures.py` defines the shared training model, model builder, and parameter builders. Use a parameter for tests that only call
optimizer methods; use a model when the test needs a forward pass or a model-based API. Scalar, complex, and
inactive parameters use the same builder. Sparse comparisons use independent dense and sparse copies.

`utils.py` contains optimizer construction, training, and assertion helpers. Shared tests use `build_optimizer()`;
focused validation tests use `load_optimizer()` directly so helper defaults do not affect invalid inputs.
`create_optimizer()` belongs in its API tests. Wrapper tests construct wrappers directly to exercise their APIs.

`recipes.py` contains named training configurations and iteration counts. Keep configurations needed for convergence
and variant coverage explicit. Scheduler expectations, loss inputs, and variant recipes live beside their tests.
Parameter preparation copies each recipe before attaching model-specific callbacks or parameter groups.

`optimizer_cases.py` derives constructor options from signatures and sparse/complex support from cached CPU steps.
Probes preserve random state and propagate unexpected failures. Optimizers needing models, distributed setup, or
special update protocols have dedicated tests instead of these probes. Remaining exclusions describe specific
validation or numerical limitations. Strict expected failures in the maximize tests document existing first-step
issues in BCOS, SGDSaI, and TAM.

`test_optimizers.py` covers shared training and optimizer interfaces. Training tests select supported dtype and foreach
combinations; interface tests run once per optimizer. `optimizers/test_<module>.py` contains focused optimizer and
algorithm-helper tests. Wrapper, gradient, parameter, and variant tests group related cases into pytest classes.

Run the repository checks with:

```shell
uv run just format
uv run just check
uv run just test
uv run coverage report -m
```

Select an optimizer or test class for a focused run:

```shell
uv run pytest tests/optimizers/test_muon.py -sv -vv
uv run pytest tests/test_optimizers.py::TestOptimizerTraining -sv -vv
```
