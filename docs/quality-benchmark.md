# Quality Benchmark

Runtime benchmarks answer "how fast does the workflow run?" Quality benchmarks
answer a different question:

> Did the system produce a correct, complete, structured, and safe answer?

JadeAgent now includes a deterministic quality benchmark:

```bash
python benchmarks/quality_compare.py --out-dir benchmarks/out --json
```

## What It Tests

The benchmark uses small tasks with objective rubrics:

- required concepts must appear;
- required sections must appear;
- forbidden claims must not appear;
- output length must stay inside a useful range.

Targets:

- `raw_baseline`: intentionally weak baseline answers.
- `jade_agent`: JadeAgent with a deterministic scripted backend.
- `langgraph`: LangGraph `StateGraph` with the same deterministic scripted
  backend, when LangGraph is installed.
- `jade_agent_jgx`: JadeAgent with the same backend plus SQLite-backed JGX
  state.

## Metrics

- `score`: weighted rubric score from 0.0 to 1.0.
- `passed`: score greater than or equal to the pass threshold.
- `pass_rate`: passed cases divided by total cases.
- `avg_score`: mean rubric score.
- `missing_terms`: required concepts not present.
- `missing_sections`: required sections not present.
- `jgx_event_count`, `jgx_snapshot_count`, `jgx_verify_ok`: state quality and
  audit evidence for the JGX target.

## Versus LangGraph

The LangGraph target uses the same scripted backend and the same prompts. That
means matching rubric scores should be expected: graph runtimes do not improve
answer quality by themselves when the model/backend output is held constant.

The distinction to look for is runtime evidence:

- `langgraph` can orchestrate the answer path;
- `jade_agent_jgx` can orchestrate and also emit JGX events, snapshots, and
  integrity verification.

For real model quality, add equivalent real-backend targets for both JadeAgent
and LangGraph, then score the outputs with the same objective rubric and an
optional LLM/human judge.

## Colab

After mounting Drive and installing JadeAgent:

```python
%cd /content/drive/MyDrive/JadeAgent
!pip install -q -e .
!python benchmarks/quality_compare.py --out-dir benchmarks/out --json
```

Show the newest report:

```python
from pathlib import Path
reports = sorted(Path("benchmarks/out").glob("quality_compare_*.md"))
print(reports[-1])
print(reports[-1].read_text())
```

## Next Level

For model quality, add a real backend target and an LLM judge. The same JSON
artifact can store both:

- objective rubric score;
- unit-test or hidden-test result;
- LLM judge score;
- human review score.

The safest portfolio claim is to report all of them separately instead of
collapsing quality into one vague number.
