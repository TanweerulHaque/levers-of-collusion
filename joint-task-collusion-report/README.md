# Looking the Other Way: Levers of Collusion in LLM Peer Review

Report and slides for the Apart Research AI Collusion Research Sprint (23–25 October 2026), Track 3.
Experiments live in [TanweerulHaque/levers-of-collusion](https://github.com/TanweerulHaque/levers-of-collusion),
built on [SALT-NLP/agent-collusion](https://github.com/SALT-NLP/agent-collusion) (branch `main` is the original code;
branches `tanweer/exp-2` to `tanweer/exp-9` each add one lever).

| File | What it is |
| --- | --- |
| `report.pdf`, `report.tex` | The report (8 pages of main text, then references and appendices) |
| `looking-the-other-way-slides.pptx` | The slide deck (15 slides, speaker notes on every slide) |
| `numbers.tex`, `tables/` | Every number and table in the report, generated; do not edit by hand |
| `figures/` | Report figures (PDF for LaTeX, PNG for slides) |
| `data/` | Per-condition summaries computed from the run records |
| `analysis/` | Scripts that read the run records in `joint-task-collusion/results/` |

## Which runs are used

Only launches laid out as `results/<pair>/<label>_<datetime>/rep<N>/run.json`. Older `run_*` folders are excluded.
The exact file list for each condition is `SETS` in `analysis/harness.py`. Runs dated before 23 October 2026 predate the sprint;
the 25-trajectory baselines (Luna→GLM, GLM→Luna, DeepSeek self-pair) and the per-term penalty runs were made during it.

## Rebuild

Each condition is scored by the code of its own branch, so `harness.py` runs once per branch, in the experiment repo.

```bash
cd ../joint-task-collusion
R=../joint-task-collusion-report
git checkout tanweer/exp-2 && python $R/analysis/harness.py exp-2 $R/data/exp2.json LG_base25 LG_base5 LG_base_oct20 GL_base25 GL_base5 DD_base DM_base DM_base+inc HM_forecast+inc LG_pen25 LG_pen5 GL_pen GL_pen+inc --cmp LG_base25:LG_pen25 --cmp GL_base25:GL_pen --cmp LG_base5:LG_base_oct20 --cmp LG_base25:GL_base25 --cmp LG_base5:LG_pen5
git checkout tanweer/exp-3 && python $R/analysis/harness.py exp-3 $R/data/exp3.json LG_forecast HM_forecast+inc --cmp LG_base25:LG_forecast --cmp LG_base5:LG_forecast
git checkout tanweer/exp-5-announcing-of-episodes-communication-rounds && python $R/analysis/harness.py exp-5 $R/data/exp5.json LG_countdown LG_countdown_pen --cmp LG_base25:LG_countdown --cmp LG_pen25:LG_countdown_pen --cmp LG_base5:LG_countdown --cmp LG_countdown:LG_countdown_pen
git checkout tanweer/exp-6-restore-full-log-link-tool && python $R/analysis/harness.py exp-6 $R/data/exp6.json LG_restore50 LG_restore110 GL_restore110 DM_restore50+inc DM_base+inc --cmp LG_base25:LG_restore50 --cmp LG_base25:LG_restore110 --cmp GL_base25:GL_restore110 --cmp DM_base+inc:DM_restore50+inc
#   plus: python analysis/restore_channel_metrics.py --runs 'results/<pair>_restore_channel_link_cost<N>/*/rep*/run.json' --json $R/data/restore_<pair>_restore_channel_link_cost<N>.json
git checkout tanweer/exp-7-cost-per-character-in-communication && python $R/analysis/harness.py exp-7 $R/data/exp7.json LG_msgcost+inc LG_base25 --cmp LG_base25:LG_msgcost+inc
#   plus: python analysis/message_cost_metrics.py --runs 'results/openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor_message_cost_rate0_005/*/rep*/run.json' --json $R/data/msgcost.json
git checkout tanweer/exp-8-sequential-verdict && python $R/analysis/harness.py exp-8 $R/data/exp8.json LG_seq GL_seq DM_seq DM_base --cmp LG_base25:LG_seq --cmp GL_base25:GL_seq --cmp DM_base:DM_seq
#   plus: python analysis/collusion_metrics.py --runs <sequential and blind launches> --json $R/data/seq_collusion.json
git checkout tanweer/exp-9-task-difficulty && python $R/analysis/harness.py exp-9 $R/data/exp9.json LG_diff GL_diff DM_diff+inc DM_base+inc --cmp LG_base25:LG_diff --cmp GL_base25:GL_diff --cmp DM_base+inc:DM_diff+inc
#   plus: python analysis/difficulty_label_metrics.py --runs '<launch>/rep*/run.json' --peer-difficulty-label true --json $R/data/diff_<pair>.json
git checkout tanweer/exp-2

cd $R
python analysis/evidence.py NAME=GLOB ...   # unsupported mutual ACCEPT for the restore and message-price runs
python analysis/extras.py                   # noise band, forecast calibration, deal phrases, totals
python make_numbers.py                      # data/numbers.json, numbers.tex, tables/
python make_figures.py                      # figures/
tectonic -X compile report.tex              # report.pdf
NODE_PATH=<node_modules with pptxgenjs, react-icons, react, react-dom, sharp> node build_deck.js <pptx skill>/scripts/apply_theme.js
```

`harness.py` calls the branch's own `analysis/compare_runs.py` (the code behind the results viewer's Compare tab) on lists
of run files, so pooled sets that span launches, such as GLM→Luna under the penalty, are scored exactly as the Compare tab
scores a folder.

## Prior work

Code written before the sprint: the original release (branch `main`); the YAML configuration, OpenRouter routing,
parallel repeats, results viewer and `compare_runs.py` (`tanweer/exp-1`, `tanweer/exp-2`); the lever code on
`tanweer/exp-3` to `tanweer/exp-9`; and every 5-trajectory lever run (9–20 October 2026). Sprint work: the 25-trajectory
baselines and penalty runs (24–25 October), the analysis in this folder, the report and the slides.
