"""One module per top-level verb: `easel <verb>` lives in `easel/commands/<verb>.py`.

A verb module is the thin argparse layer (`register(subparsers)` plus its handlers); the logic stays in the
domain modules it calls. The conventions are at the top of `easel/cli.py`.
"""
