# mesh_taskpack

`mesh_taskpack` is a small standard-library Python package generated
by a JadeAgent mesh workflow. It models tasks, dependencies, owners,
status transitions, JSON persistence, markdown export, and a tiny CLI.

## Quick Start

```bash
python -m mesh_taskpack.cli --file demo.json demo
python -m mesh_taskpack.cli --file demo.json summary
```

## Development

```bash
python -m unittest discover -s tests -p "test_*.py"
```
