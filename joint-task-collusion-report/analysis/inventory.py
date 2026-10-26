import json, glob, os, sys
R = "/Users/tanweerulhaque/Desktop/NTU/NAIS_hackathon/joint-task-collusion/results"
keys = ["reward_penalty","reward_scheme","forecast","forecast_position","episode_announcement","restore_channel_tool",
        "restore_channel_cost_percent","message_cost","message_cost_rate","verdict_order","peer_difficulty_label",
        "max_rounds","char_limit","task_sequence_record","start_index","repeat_index","pair_count","temperature"]
for launch in sorted(glob.glob(f"{R}/*/*/")):
    reps = sorted(glob.glob(launch + "rep*/run.json"))
    if not reps: continue
    rows = []
    for p in reps:
        r = json.load(open(p)); c = r.get("run_config", {})
        res = r.get("results") or []
        planned = c.get("pair_count") or len(c.get("throttle_policy_by_episode") or []) or len(res)
        seq = c.get("task_sequence_record") or c.get("task_sequence") or c.get("sequence_id")
        rows.append((os.path.basename(os.path.dirname(p)), len(res), planned, c.get("sequence_name") or c.get("task_sequence_name"), c.get("start_index"), os.path.exists(os.path.dirname(p)+"/agreement.csv")))
    c0 = json.load(open(reps[0])).get("run_config", {})
    print("==", launch.replace(R+"/",""))
    print("   models:", c0.get("models"), "| flags:", {k: c0.get(k) for k in keys if k in c0})
    inc = [f"{n}:{o}/{pl}" for n,o,pl,*_ in rows if o < pl]
    print(f"   reps={len(rows)} complete={sum(1 for _,o,pl,*__ in rows if o>=pl)} incomplete={inc} judge_csv={sum(1 for *_,j in rows if j)}")
