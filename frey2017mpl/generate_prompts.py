import pandas as pd
import os
import json
import math
import numpy as np

base_path = "basel_berlin_data"
folders = ["main"]

output_prompts = []

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

def format_number(num):
    if num == int(num):
        return f"{int(num)}"
    else:
        return f"{num:.1f}"

for folder in folders:
    mpl_path = os.path.join(base_path, folder, "mpl", "mpl.csv")
    mpl_problems_path = os.path.join(base_path, folder, "mpl", "mplProblems.csv")
    participants_path = os.path.join(base_path, folder, "participants", "participants.csv")

    mpl_data = pd.read_csv(mpl_path)
    mpl_problems = pd.read_csv(mpl_problems_path)
    participants = pd.read_csv(participants_path)

    merged_data = pd.concat([mpl_data.reset_index(drop=True), mpl_problems.reset_index(drop=True)], axis=1)
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
    grouped = merged_data.groupby(['dp', 'decision'])
    merged_data['A_out1'] = grouped['A_out1'].transform('first')
    merged_data['A_out2'] = grouped['A_out2'].transform('first')
    merged_data['A_p1'] = grouped['A_p1'].transform('first')
    merged_data['A_p2'] = grouped['A_p2'].transform('first')
    merged_data['B_out1'] = grouped['B_out1'].transform('first')
    merged_data['B_out2'] = grouped['B_out2'].transform('first')
    merged_data['B_p1'] = grouped['B_p1'].transform('first')
    merged_data['B_p2'] = grouped['B_p2'].transform('first')

    for participant_id in merged_data["partid"].unique():
        participant_data = merged_data[merged_data["partid"] == participant_id]
        meta_row = participants[participants["partid"] == participant_id]

        trials_text = []

        session_text = (
            "You will be presented with several pairs of lotteries in each trial. Each lottery offers specific chances of winning or losing points.\n"
            "Your task is to choose between lottery A and lottery B. Each choice affects your potential earnings.\n"
            "No immediate feedback will be provided regarding the outcomes of your choices.\n"
        )

        for trial_number, trial_data in participant_data.groupby("dp"):

            trial_text = [f"List {int(trial_number)}:\n"]

            for trial in trial_data.itertuples():
                if any(math.isnan(x) for x in [trial.A_p1, trial.A_p2, trial.B_p1, trial.B_p2]):
                    continue

                decision_text = (
                    f"Lottery A offers a {trial.A_p1 * 100:.0f}% chance to win {format_number(trial.A_out1)} points and a "
                    f"{trial.A_p2 * 100:.0f}% chance to lose {format_number(trial.A_out2)} points. "
                    f"Lottery B offers a {trial.B_p1 * 100:.0f}% chance to win {format_number(trial.B_out1)} points and a "
                    f"{trial.B_p2 * 100:.0f}% chance to lose {format_number(trial.B_out2)} points. "
                    f"You chose lottery <<{'A' if trial.choice == 0 else 'B'}>>.\n"
                )
                trial_text.append(decision_text)

            if len(trial_text) > 1:
                print(f"Completed Trial {int(trial_number)} with {len(trial_text) - 1} decisions.")
                trials_text.append("\n".join(trial_text))
            else:
                print(f"Skipping Trial {int(trial_number)} due to no valid decisions.")

        participant_text = session_text + "\n\n" + "\n\n".join(trials_text)
        meta_info = {}
        if not meta_row.empty:
            for field in ["sex", "age", "location"]:
                if field in meta_row.columns and not pd.isnull(meta_row.iloc[0][field]):
                    meta_info[field] = meta_row.iloc[0][field]
        prompt = {
            "participant": f"{participant_id}",
            "experiment": "MPL",
            "text": participant_text,
        }
        if meta_info:
            prompt.update(meta_info)
        output_prompts.append(prompt)

output_path = "prompts.jsonl"

with open(output_path, "w") as f:
    for prompt in output_prompts:
        prompt = {k: convert_to_builtin_type(v) for k, v in prompt.items()}
        f.write(json.dumps(prompt) + "\n")