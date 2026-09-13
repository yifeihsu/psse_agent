# IEEE57 numerical testbeds

The balanced corpus, logical-CB topology simulator, and three-phase OpenDSS
testbed are reproducible from source. Install the CPU numerical dependencies with
`python -m pip install -r psse_env/requirements-ieee57.txt`.

Large corpora and generated models remain local under ignored `output/`,
`generated/`, and `tmp/` directories. They are not bundled with the Git branch.
The packages and validation scripts generate their own inputs; historical reports
may reference local artifacts from the original runs.

The compact `coupler_nlm_feasibility_20260912.json` result records a standalone
experiment, not a production NLM implementation or a training dataset. Reproduce
it from a repository checkout, choosing a new output filename:

```sh
python scripts/probe_exact_coupler_nlm.py --output output/coupler_nlm_new.json
```

See the [method review](../../docs/ieee57_topology_method_review_20260912.md)
for assumptions and the [logical testbed](../../logical_topology/README.md)
for generation and correction interfaces.
