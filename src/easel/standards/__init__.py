"""Usage standards: how each resource should be used (weight, CFG, steps, sampler, ...), and from which evidence.

Every value is kept per source in `<home>/standards/cards/<sha16>.yaml` (schema easel/usage-card/1), so the
sources never overwrite each other, and one fixed precedence picks the value a recipe uses:

    manual      the person said so (`easel standards set`)                       strongest
    experience  the current verdicts on our own runs (counts from 3 verdicts on)
    author      the creator's usage text and the facts the file carries
    references  example images the person collected, which name the resource
    prior       the family defaults in families/<id>.yaml                        weakest

Nothing is averaged across sources: the strongest source that has the field wins, and `easel standards
explain <sha16>` shows all five side by side with their evidence. Modules:
- `author`: reading numbers out of an author's usage text;
- `experience`: what our own runs say about each resource (runs x current verdicts);
- `sources`: gathering every source's values for every resource;
- `cards`: precedence, the card files, build, set, unset, explain and report.
The verbs are the argparse layer in `easel/commands/standards.py`.
"""
