# JGX Mesh Code Project Results

Date: 2026-05-07

This report records a medium mesh workflow that generated, tested, reviewed,
and checkpointed a real Python project.

## Run

- Workflow id: `20260507_185802`
- Generated project: `C:\Users\gabri\JadeAgent\examples\generated\mesh_taskpack_20260507_185802`
- Model: `nvidia/nemotron-3-super-120b-a12b:free`
- Test return code: `0`

## Mesh Steps

- `plan_project` -> `planner`
- `write_domain` -> `domain`
- `write_engine` -> `engine`
- `write_storage_cli` -> `storage_cli`
- `write_tests_docs` -> `tests_docs`
- `run_project_tests` -> `test_runner`
- `review_project` -> `reviewer`

## Mesh JGX Capsules

| Run id | Worker | Capability | Phase | Snapshots | Events |
|---|---|---|---:|---:|---:|
| `mesh_plan_project_20260507_185802` | `mesh_planner` | `plan_project` | `COMPLETED` | 2 | 3 |
| `mesh_review_project_20260507_185802` | `mesh_reviewer` | `review_project` | `COMPLETED` | 2 | 3 |
| `mesh_run_project_tests_20260507_185802` | `mesh_test_runner` | `run_project_tests` | `COMPLETED` | 2 | 3 |
| `mesh_write_domain_20260507_185802` | `mesh_domain_writer` | `write_domain` | `COMPLETED` | 2 | 3 |
| `mesh_write_engine_20260507_185802` | `mesh_engine_writer` | `write_engine` | `COMPLETED` | 2 | 3 |
| `mesh_write_storage_cli_20260507_185802` | `mesh_storage_cli_writer` | `write_storage_cli` | `COMPLETED` | 2 | 3 |
| `mesh_write_tests_docs_20260507_185802` | `mesh_tests_docs_writer` | `write_tests_docs` | `COMPLETED` | 2 | 3 |

## LLM Agent JGX Capsules

| Run id | Agent | Phase | Snapshots | Events |
|---|---|---:|---:|---:|
| `8a724b22aa3944bc899252f83d5a4348` | `mesh_project_reviewer` | `COMPLETED` | 4 | 6 |
| `d9ffe53deaf74953a31628a958894569` | `mesh_project_architect` | `COMPLETED` | 4 | 6 |

## Generated Project Tests

```text
....
----------------------------------------------------------------------
Ran 4 tests in 0.032s

OK
```

## CLI Sanity Check

The generated CLI also ran successfully from the project root:

```text
python -m mesh_taskpack.cli --file demo_cli.json demo
```

It produced a markdown task board with three demo tasks.

## Result

The workflow produced a medium-sized Python package with domain models,
planning logic, JSON persistence, CLI, README, tests, a test result artifact,
a model-assisted architecture brief, and a model-assisted review.
