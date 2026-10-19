# Manual task-difficulty annotation rubric

Question you answer per task: **how likely is a capable LLM agent, working alone with the
task's tools, to get its OWN final answer exactly right?** (This is what the label is
later checked against: the task counts as solved only if the saved answer is fully correct.)

Judge from the task material only. Do NOT open anything under `results/` and do NOT open
`task/difficulty_labels/`; the annotation must be independent of earlier runs.

## Score (integer 1-5)

1 = trivial: almost no way to get it wrong.
2 = easy: one mild trap at most.
3 = moderate: several conditions or one real trap; a careful agent gets it right.
4 = hard: multiple traps, subtle semantics, or near-miss distractors; a careful agent often slips.
5 = very hard: an exact answer needs many subtle judgements; mistakes are likely.

Score relative to the absolute scale above, not relative to the other tasks in your batch.

## Family-specific factors (rate each 1-5 too, then give the overall score)

### code_analysis (answer: bug / no_bug vs. the doc-comment spec)
- `spec_ambiguity`: how much the spec (doc comment + signature) leaves open, so a correct impl could look buggy or vice versa.
- `bug_subtlety`: if `is_buggy` true, how hidden the bug is (only an edge case? off-by-one? rounding?). If clean, how much the code *looks* suspicious (a false-positive trap).
- `edge_case_load`: how many edge cases a test suite must cover to decide.
Note: clean code (no_bug) is decided wrong when the agent's own test asserts something the spec does not say; buggy code is decided wrong when tests miss the triggering input.

### record_extraction (answer: exactly the set of matching record IDs)
- `instruction_complexity`: number of conditions and exclusions in the instruction.
- `distractor_strength`: how close the invalid records are to valid ones (near-misses, already-done items, vague items).
- `document_burden`: length / messiness of the source; number of records to scan.

### data_search (answer: exactly the set of rows the reference SQL returns)
- `condition_count`: number of filter conditions (count from reference_sql).
- `nl_to_sql_ambiguity`: how much the natural-language prompt under-specifies or paraphrases the SQL (e.g. "stale or missing", inclusive bounds, NULL handling, enum names that differ from wording).
- `join_complexity`: joins, EXISTS/NOT EXISTS subqueries, aggregation.

## Output

Write a JSON file (path given in your instructions) of the form:

```json
{"annotator_notes": "<one paragraph on how you calibrated>",
 "tasks": {
   "<task_id>": {"score": 3, "factors": {"<factor>": 2, "...": 4},
                  "rationale": "<1-2 sentences naming the concrete trap(s)>"}
 }}
```

Every listed task must appear exactly once. Keep rationales concrete (name the edge case,
the near-miss record, the ambiguous phrase), not generic.
