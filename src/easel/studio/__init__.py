"""The studio: recipes, renders through the scheduler, the run ledger, GPU jobs, the server's lifecycle, doctor.

What each module answers (the verbs are the thin argparse layer in `easel/commands/`):
- `common`: the instance's folders and policy, the adopted-workspace refusal, the error type every verb maps to
  its exit code;
- `recipe`: `easel recipe propose|revise|like|show|list|approve` (a recipe is a file, `recipes/<rcp>.json`);
- `render`: `easel run <rcp>`: recipe -> resolved spec -> graph -> scheduler -> ComfyUI -> `runs/<run-id>/`;
- `ledger`: `easel runs list|show|compare|reviews|review`, including the current verdict per output;
- `jobs`: `easel jobs submit|status|cancel|run`, the queue a GPU job waits in until the machine has room;
- `lifecycle`: `easel comfy start|stop|status` and the "is a server there" check the runner shares;
- `doctor`: `easel doctor`, one `ok|warn|FAIL` line per check, then `next:`;
- `stage`: `easel next`, the stage, the runbook to read and the next command.

Nothing here opens, decodes or describes an image: outputs are written as the server sends them and probed
header-only (`_common.image_meta`). GPU work never starts from a dry run, and never on an adopted workspace
(`policy.render = "legacy-runner"`), where the host's own runner renders.
"""
