import pandas as pd
import os
import json

base_path = "basel_berlin_data"
folders = ["main"]
import numpy as np

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
    cct_path = os.path.join(base_path, folder, "cct", "cct.csv")
    cct_data = pd.read_csv(cct_path)
    participants_path = os.path.join(base_path, folder, "participants", "participants.csv")
    participants = pd.read_csv(participants_path)

    for participant_id in cct_data["partid"].unique():

        participant_data = cct_data[cct_data["partid"] == participant_id]
        meta_row = participants[participants["partid"] == participant_id]

        decks_text = []

        session_text = (
            "You are about to play a card game where you can win or lose points that are worth money. "
            "Each round starts with 32 face-down cards. Some are gain cards, adding points, and others are loss cards, subtracting points. "
            "You will know how many gain and loss cards are in the deck and the points they add or subtract, but not their positions. "
            "Decide how many cards you want to turn over. For each gain card, points are added, and the process continues. "
            "If you reveal a loss card, the corresponding points are subtracted, and the round ends immediately.\n"
        )

        for deck_number, deck_data in participant_data.groupby("r_trialnum"):

            gain_value = deck_data["r_winvalue"].iloc[0]
            loss_value = deck_data["r_lossvalue"].iloc[0]
            loss_cards = int(deck_data["r_lossnum"].iloc[0])
            cards_chosen = int(deck_data["r_cardschosen"].iloc[0])
            censored = int(deck_data["r_censored"].iloc[0])
            final_score = int(deck_data["r_score"].iloc[0])

            initial_gain_cards = 32 - loss_cards

            remaining_gain_cards = initial_gain_cards
            remaining_loss_cards = loss_cards
            accumulated_points = 0
            deck_decisions = []

            deck_text = [f"Deck {int(deck_number)}:"]

            if cards_chosen == 0:
                deck_decisions.append("The deck contains "
                                      f"{remaining_gain_cards} gain cards worth {format_number(gain_value)} points each and "
                                      f"{remaining_loss_cards} loss cards worth {format_number(loss_value)} points. "
                                      "You decided to <<not play>>.")
            else:
                for draw in range(1, cards_chosen + 1):
                    if censored > 0 and draw == censored:
                        accumulated_points -= loss_value
                        deck_decisions.append(
                            f"The deck contains {remaining_gain_cards} gain cards worth {format_number(gain_value)} points each and "
                            f"{remaining_loss_cards} loss cards worth {format_number(loss_value)} points. "
                            f"You decided to <<draw>> and found a loss card. You lost {format_number(loss_value)} points. "
                            f"The round is over. Your accumulated points are {format_number(accumulated_points)}."
                        )
                        break

                    else:
                        accumulated_points += gain_value
                        deck_decisions.append(
                            f"The deck contains {remaining_gain_cards} gain cards worth {format_number(gain_value)} points each and "
                            f"{remaining_loss_cards} loss cards worth {format_number(loss_value)} points. "
                            f"You decided to <<draw>> and found a gain card. You gained {format_number(gain_value)} points. "
                            f"Your accumulated points are {format_number(accumulated_points)}."
                        )

                        remaining_gain_cards -= 1

                if censored == 0:
                    deck_decisions.append(
                        f"The deck contains {remaining_gain_cards} gain cards worth {format_number(gain_value)} points each and "
                        f"{remaining_loss_cards} loss cards worth {format_number(loss_value)} points. "
                        f"You decided to <<stop>>. Your accumulated points are {format_number(final_score)}."
                    )

            if deck_decisions:
                decks_text.append("\n".join(deck_text + deck_decisions))

        participant_text = session_text + "\n\n".join(decks_text)

        meta_info = {}
        if not meta_row.empty:
            for field in ["sex", "age", "location"]:
                if field in meta_row.columns and not pd.isnull(meta_row.iloc[0][field]):
                    meta_info[field] = meta_row.iloc[0][field]

        prompt = {
            "participant": f"{participant_id}",
            "experiment": "CCT",
            "text": participant_text,
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
