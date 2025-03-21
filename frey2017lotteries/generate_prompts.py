import pandas as pd
import os
import json
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


for folder in folders:
    lotteries_path = os.path.join(base_path, folder, "lotteries", "lotteries.csv")
    participants_path = os.path.join(base_path, folder, "participants", "participants.csv")

    lotteries_data = pd.read_csv(lotteries_path)
    participants = pd.read_csv(participants_path)


    def format_number(num):
        return f"{int(num)}" if num == int(num) else f"{num:.1f}"


    for participant_id in lotteries_data["partid"].unique():

        participant_data = lotteries_data[lotteries_data["partid"] == participant_id]
        meta_row = participants[participants["partid"] == participant_id]

        gambles_text = []

        session_text = (
            "You will make decisions between two options, each representing a lottery with varying outcomes and probabilities. "
            "For each lottery, you will be shown the potential gains, losses, and the probabilities of winning or losing. "
            "Each point earned corresponds to a monetary value. "
            "No immediate feedback will be provided on the outcomes of your choices.\n"
        )

        for gamble_number, gamble_data in participant_data.groupby("Dec_ID"):

            gamble_text = [f"Gamble {int(gamble_number)}:"]

            for trial in gamble_data.itertuples():
                lottery_A = (
                    f"Lottery A: Gain {format_number(trial.X1)} points with a {trial.PX1:.0f}% chance or "
                    f"gain {format_number(trial.X2)} points with a {100 - trial.PX1:.0f}% chance."
                )
                lottery_B = (
                    f"Lottery B: Gain {format_number(trial.Z1)} points with a {trial.PZ1:.0f}% chance or "
                    f"gain {format_number(trial.Z2)} points with a {100 - trial.PZ1:.0f}% chance."
                )
                decision_text = (
                    f"{lottery_A} {lottery_B} You chose <<{'A' if trial.Decision_X == 1 else 'B'}>>."
                )

                gamble_text.append(decision_text)

            gambles_text.append("\n".join(gamble_text))

        participant_text = session_text + "\n\n" + "\n\n".join(gambles_text)
        meta_info = {}
        if not meta_row.empty:
            for field in ["sex", "age", "location"]:
                if field in meta_row.columns and not pd.isnull(meta_row.iloc[0][field]):
                    meta_info[field] = meta_row.iloc[0][field]
        prompt = {
            "participant": f"{participant_id}",
            "experiment": "Lotteries",
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