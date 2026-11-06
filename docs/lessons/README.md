# Lessons

A lesson is one finding, the evidence for it and the rule it implies: "the demo_robot LoRA at 0.8 keeps the visor
in 3 of 4 renders" (finding), "run-20300101-001: outputs 1, 2 and 4 kept, output 3 rejected" (evidence), "start
the demo_robot LoRA at 0.8 on Illustrious checkpoints" (rule). A verdict says what happened to one image; a
lesson says what to do next time, so a later session, or a weaker model, applies it without learning it again.
The step-by-step runbook for agents is [07-lessons.md](../runbooks/07-lessons.md).

## Recording a lesson

```
easel lessons add --family <family> --finding "<finding>" --evidence "<evidence>" --rule "<rule>"
easel lessons add --finding "<finding>" --evidence "<evidence>" --rule "<rule>" --source civitai
easel lessons list --family <family>
easel lessons render
```

- Each `add` appends one line to `<home>/lessons/entries.jsonl` (format `easel/lessons-entry/1`, see
  [formats](../reference/formats.md)) with the next id `les-YYYYMMDD-NNN`. Leave `--family` out for a lesson that
  holds for every family.
- Write the operator's words. A finding of your own needs `--assistant`: it is stored under the second source
  (`policy.second_source_name`) and never stands for the operator.
- `--source` says where the finding came from: `review` (verdicts), `run`, `search`, `manual` (the default) or
  `civitai` for feedback the operator copied from a model page by hand. No tool reads comments: that site's
  terms allow automation only through its public API ([civitai rules](../reference/civitai-rules.md)).
- A lesson is never edited or deleted. A correction is a new lesson whose evidence names the one it corrects.

## The pages

`easel lessons render` writes one page per family, `<family>.md`, and `general.md` for lessons without a family,
each with the table Finding | Evidence | Rule | Recorded in. The pages go into `<home>/lessons/` by default and
carry a "generated, do not edit" marker; a page without that marker is never overwritten.

Pages in this folder of the repository are lessons contributed back: render them with
`easel lessons render --out <dir>`, scan them with `easel privacy scan --public <dir>` (no subject names, ids,
paths or account names may leave your machine), and open a pull request with the `pitfall` or
`resource-knowledge` label. Prose pages written by hand, such as [privacy.md](privacy.md), sit beside the
generated ones and are never replaced by `render`.
