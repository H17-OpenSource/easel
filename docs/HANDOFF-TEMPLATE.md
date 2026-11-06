# Hand-off template

Use this shape for the one message that ends a session, whether the work is done, waiting on someone, or
stuck. Fill every `<...>` from commands and the journal, never from memory. Keep it short: the next reader
(the operator, a maintainer, or the next agent session) should know in one minute where things stand and
what to run first.

Rules:
- Every number, id and state comes from a command you ran in this session; give the command beside it.
- Dates and times come from `date`; the clock is the only source.
- No image descriptions. A judgement the operator has not given is written as an open question, not filled.
- No private values in a hand-off that may leave the machine: no absolute paths (write `<home>`,
  `<models_root>`), no names, no tokens.
- One message, then stop. Do not keep working after the hand-off unless the reader answers.

## Template

```
# Hand-off <date from the clock> <time> easel <instance name>

## Now
stage: <the line `easel next --stage` prints>
next: <the line `easel next` prints after "next:">
running: <job or run ids and the state read once, or "nothing">
waiting on: <operator yes | image judgement | maintainer | nothing>, for <what, in one line>

## Done this session
- `<command>` -> `<its last line>` (exit <n>)
- ...

## Not done, and why
- <step> - <reason: planned verb | gate refused | waiting on someone | out of time>

## Open questions for the operator
- <question in one sentence>. Options: a) <...> b) <...>. Until answered: <wait | the default the runbook names>

## Pitfalls recorded
- <symptom> -> <rule>; row added to `<runbook file>` section 6, dated <date>; journal line at <time>

## STUCK notes
<paste the blocks from `easel journal report`, or "none">

## First command for the next session
`<exact command>`
```

A job you submitted keeps running after you stop: give its id under "running" so the next session reads its
state once with `easel jobs status <job-id>` instead of submitting it again. The journal of the day is the
long version: see [JOURNAL.md](JOURNAL.md).
