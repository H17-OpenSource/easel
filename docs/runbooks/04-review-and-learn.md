# 04-review-and-learn: the operator's verdicts become experience

Stage map: [README.md](README.md).

## 1. Purpose and who decides

Record the operator's verdict on each output of a run, then fold the verdicts into the usage standards, so
the next `recipe propose` prefers what the operator kept.

- **The operator is the only judge.** Every verdict (keep, maybe, reject) and every note is theirs; you
  transcribe their words with `--note "<their words>"` and never add one of your own.
- **You:** never open, decode or describe an output. If `policy.reviewer` allows an assistant read of an sfw
  image, label it "assistant's read, not the verdict" and keep it out of the verdict records.
- easel's own review page is planned (session-2). Until it lands the operator gives verdicts in chat or the
  terminal and you record them; on an adopted workspace the operator reviews in the host's own review page
  ([90-adopt-existing.md](90-adopt-existing.md)).

## 2. Entry conditions

| command | literal line |
|---|---|
| `easel next --stage` | `04-review-and-learn` |

A step marked planned is not built: report it once and stop at that stage.

## 3. Commands

Read first; every write carries the operator's words.

```
easel runs list
easel runs show <run-id>
easel runs review <run-id> --verdict keep --output <n> --note "<the operator's words>"
easel runs review <run-id> --verdict reject --note "<the operator's words>"
easel runs reviews <run-id> --current
easel standards build
easel standards explain <sha16>
easel standards report
easel note <id> "<the operator's words>"
```

- `runs show <run-id>` numbers the outputs the way `--output` takes them. Without `--output`, a verdict
  covers the whole run.
- The latest verdict per output counts (`runs reviews --current`): a changed mind replaces the verdict; it
  never adds a second one.
- `standards build` turns current verdicts into experience: from 3 reviewed images on, the keep rate, the
  weight that produced keepers and the settings of kept runs outrank the creator's advice.
- A lesson the operator states (what to do next time, with its evidence) is recorded through
  [07-lessons.md](07-lessons.md).

## 4. Expected output

`easel runs show run-20300101-001` on the demo:

```
run-20300101-001  ok  recipe rcp-20300101-001  job job-20300101-001  seed 1001  832x1216  28 steps CFG 5.5 euler_ancestral/normal
  output 1: outputs/001.png  8x8  seed 1001  keep (operator)
  output 2: outputs/002.png  8x8  seed 1002  keep (operator)
  output 3: outputs/003.png  8x8  seed 1003  reject (operator) - the visor split into two eyes
  output 4: outputs/004.png  8x8  seed 1004  keep (operator) - on a second look the vest is orange
  manifest: runs/run-20300101-001/manifest.json
```

`easel runs review run-20300101-001 --verdict keep --output 3 --note "<the operator's words>"`:

```
recorded keep on run-20300101-001 output 3 (operator)
next: easel runs reviews --current, then easel standards build
```

`easel runs reviews run-20300101-001 --current` after that verdict (output 3's earlier reject is replaced):

```
run-20300101-001  output 1   keep    operator  2030-01-01T12:00:00+00:00
run-20300101-001  output 2   keep    operator  2030-01-01T12:01:00+00:00
run-20300101-001  output 3   keep    operator  2030-01-01T00:00:00+00:00  <the operator's words>
run-20300101-001  output 4   keep    operator  2030-01-01T12:09:00+00:00  on a second look the vest is orange
```

`easel standards build`:

```
standards build: 4 card(s) written, 0 unchanged, under standards/cards/
values by source: manual 1, experience 7, author 7, references 4, prior 21 (manual kept from the cards: 1)
evidence: 1 run(s) with 4 current verdict(s); 1 reference(s), 1 resource mention(s) linked; experience counts from 3 verdicts on
```

`easel standards explain cf56db17414bcb55` (first lines):

```
cf56db17414bcb55 demo_robot-v1.safetensors (lora, family illustrious, class sfw)
card: standards/cards/cf56db17414bcb55.yaml (built 2030-01-01T00:00:00+00:00)
precedence: manual > experience > author > references > prior; experience counts from 3 verdicts on; the first source that counts wins
weight: 0.8 from manual
  manual      0.8                       set by the person
  experience  0.8 (n=4)                 4 kept / 4 reviewed
  author      0.7-0.9                   lora.recommended_weight
  references  -
  prior       0.8                       families/illustrious.yaml inference.lora_default_weight
```

Match the shape, not the numbers (see [README.md](README.md)).

## 5. Gate and operator question

1. Every output of the run has a current verdict from the operator (`easel runs reviews <run-id> --current`).
2. `easel standards build` ran after the last verdict.
3. Any lesson the operator stated is recorded with its evidence ([07-lessons.md](07-lessons.md)).

Ask once, then wait:

> Run `<run-id>` (`<rcp>`) has `<n>` outputs. Please give a verdict per output (keep, maybe or reject, and a
> note if you like), or one verdict for the whole run.

## 6. Symptom, cause, action

| symptom | cause | action |
|---|---|---|
| two verdicts seem to exist for one output | an earlier verdict was replaced | only the latest counts; `runs reviews <run-id> --current` shows it |
| `standards explain` still shows the author's numbers | fewer than 3 reviewed images use that resource | expected; more verdicts are needed before experience wins |
| a verdict is wanted but the operator has not given one | nobody may judge for them | ask; never fill it in |
| the review page's verb prints `planned: session-2 (...)` | easel's review page is not built yet | report it once; take verdicts in chat or on the host's own review page |
| the operator quotes feedback from a public post | comments are not read by any tool | record it as a lesson in their words ([07-lessons.md](07-lessons.md), `--source civitai`) |
| an output is suspected non-sfw | a rating question, not a quality verdict | do not open it; tell the operator and record their rating |

Not in the table: follow [docs/STUCK.md](../STUCK.md).

## 7. Record

```
easel journal note "04-review-and-learn: <run-id> <k> keep, <m> maybe, <r> reject; standards rebuilt"
```

A new pitfall: add a row to the table in section 6, dated from `date`, and a journal line, in the same session.
Stuck: `easel journal stuck "<one line>"`, then [docs/STUCK.md](../STUCK.md).
