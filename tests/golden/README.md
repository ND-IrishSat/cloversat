# Golden reference vectors `[LATER]`

Input/output pairs captured from `legacy/` code (or trusted external tools)
so ports can be checked against known-good numbers. One subfolder per ported
module, e.g. `golden/attitude_filter/`, `golden/bcross_law/`.

Record the legacy source path and any known legacy bug (see `# TODO(bug):`
comments) next to each vector set, so a deliberate fix isn't mistaken for a
regression.
