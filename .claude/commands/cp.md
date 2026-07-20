---
description: Commit all current changes and push
---

Commit and push the current changes:

1. Run `git status` and `git diff` (staged + unstaged) to see what changed.
2. Stage everything with `git add -A`.
3. Commit with a concise message summarizing the changes (follow this repo's commit style; end the message with the `Co-Authored-By` trailer).
4. If on the default branch, this repo is fine to push directly; otherwise push the current branch. Run `git push` (use `-u origin <branch>` if the branch has no upstream yet).
5. Report the commit hash and confirm the push succeeded.

If there are no changes to commit, say so and stop.
