**Do not delete this file!!!**

* Shall we try to refactor attention so that we inherit from a base class? Is it worth the faff?
Stemmed from this conversation:


Would it make sense for swa to be into attention.py? For tidiness...

Good question. Let me check the sizes and overlap:

 wc -l /Users/gminneci/Code/Mantile/backend/layers/attention.py /Users/gminneci/Code/Mantile/backend/layers/swa.py
~1000 lines combined. Let me check how much code is duplicated:

Read swa.py, lines 1 to 80

There's significant structural similarity. Here's my take:

Pros of merging:

Tidier - one place for all attention variants
Easier to share helper methods (effective span calc, etc.)
Less duplication in __init__.py exports
Cons of merging:

1000+ line file could feel unwieldy
SWA has distinct logic (window, sinks) that adds complexity
Harder to diff/review changes to one variant
My recommendation: Keep them separate but consider a middle ground:

Option A (current): Keep attention.py and swa.py separate - clear separation, each file is focused

Option B (refactor): Create a base AttentionBase class with shared projection/parallelism logic, then both AttentionLayer and SlidingWindowAttentionLayer inherit from it. Could live in one file or stay split.

Given that MoE also has its own file (~400 lines), the pattern of "one file per layer variant" is consistent. I'd lean toward keeping them separate unless you anticipate more attention variants (flash, linear, etc.) that would benefit from a shared base class.

Your call - want me to merge them, or leave as-is?

* add gated MLP in MoE