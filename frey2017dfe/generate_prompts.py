import pandas as pd
import os
import json
import numpy as np

base_path = "basel_berlin_data"
folders = ["main"]

output_prompts = []


def format_number(num):
    return f"{int(num)}" if num == int(num) else f"{num:.1f}"


def convert_to_builtin_type(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


for folder in folders:
    dfe_samples_path = os.path.join(base_path, folder, "dfe", "dfe_samples.csv")

    dfe_samples = pd.read_csv(dfe_samples_path)
    participants_path = os.path.join(base_path, folder, "participants", "participants.csv")
    participants = pd.read_csv(participants_path)

    for participant_id in dfe_samples["partid"].unique():

        participant_data = dfe_samples[dfe_samples["partid"] == participant_id]
        meta_row = participants[participants["partid"] == participant_id]

        trials_text = []
        all_rts = []

        session_text = (
            "You are about to make decisions based on experience. In this task, you will repeatedly draw points from"
            "two boxes, labeled A and B. "
            "Each box contains a mix of points with different values and chances of being drawn. Your goal is to learn about these chances and decide "
            "which box is likely to give you the highest reward.\n"
            "Every time you draw from a box, you will see the points you sampled, but you won’t earn a reward immediately. "
            "Once you decide to stop drawing, you will choose either Box A or Box B for your final draw, which will determine your reward."
        )

        for trial_number, trial_data in participant_data.groupby("gamble_ind"):

            trial_text = [f"Trial {int(trial_number)}:"]

            for trial in trial_data.itertuples():
                decision_text = (
                    f"You decided to <<sample from {'A' if trial.sample_opt == 'A' else 'B'}>>. "
                    f"You observed an outcome of {format_number(trial.sample_out)} points."
                )
                trial_text.append(decision_text)
                all_rts.append(int(trial.sample_rts) if not pd.isna(trial.sample_rts) else None)

            final_decision = trial_data.iloc[-1]
            final_choice = "A" if final_decision.decision == "A" else "B"
            trial_text.append(
                f"You decided to <<choose {final_choice}>> based on your observations. "
            )

            trials_text.append("\n".join(trial_text))

        participant_text = session_text + "\n\n" + "\n\n".join(trials_text)
        meta_info = {}
        if not meta_row.empty:
            for field in ["sex", "age", "location"]:
                if field in meta_row.columns and not pd.isnull(meta_row.iloc[0][field]):
                    meta_info[field] = meta_row.iloc[0][field]

        prompt = {
            "participant": f"{participant_id}",
            "experiment": "Decisions From Experience",
            "text": participant_text,
            "RTs": all_rts,
        }

        if meta_info:
            print(meta_info)
            prompt.update(meta_info)

        output_prompts.append(prompt)

output_path = "prompts.jsonl"
with open(output_path, "w") as f:
    for prompt in output_prompts:
        prompt = {k: convert_to_builtin_type(v) for k, v in prompt.items()}
        f.write(json.dumps(prompt) + "\n")
