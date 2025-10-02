import numpy as np
import pandas as pd
import random
import xgboost as xgb
from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
role_to_idx = {r: i for i, r in enumerate(roles)}

champion_data = pd.read_csv("COMBINED_MATCHSTATS-15.13.csv", low_memory=False).dropna(subset=["championName", "role", "win"])
champion_to_idx = {name: i for i, name in enumerate(champion_data["championName"].unique())}
idx_to_champion = {i: name for name, i in champion_to_idx.items()}
all_champions = list(champion_to_idx.keys())

synergy_score = np.load("synergy_score-15.13.npy")
counter_score = np.load("counter_score-15.13.npy")

model = xgb.XGBClassifier()
model.load_model("xgb_champion_model-15.13.json")

def average_synergy(team):
    pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]
    return np.mean([synergy_score[team[i]][team[j]] for i, j in pairs])

def average_counter(offense, defense):
    return np.mean([counter_score[o][d] for o in offense for d in defense])

def build_feature_vectors(blue_team, red_team):
    blue_ids = [champion_to_idx[c] for c, r in blue_team]
    red_ids = [champion_to_idx[c] for c, r in red_team]
    blue_roles = [role_to_idx[r] for c, r in blue_team]
    red_roles = [role_to_idx[r] for c, r in red_team]

    blue_synergy = average_synergy(blue_ids)
    red_synergy = average_synergy(red_ids)
    blue_vs_red = average_counter(blue_ids, red_ids)
    red_vs_blue = average_counter(red_ids, blue_ids)

    lane_counters = [0] * 5
    for role_id in range(5):
        blue_champ = [c for c, r in zip(blue_ids, blue_roles) if r == role_id]
        red_champ = [c for c, r in zip(red_ids, red_roles) if r == role_id]
        if blue_champ and red_champ:
            lane_counters[role_id] = counter_score[blue_champ[0]][red_champ[0]]

    features1 = []
    for cid, rid in zip(blue_ids + red_ids, blue_roles + red_roles):
        features1.extend([cid, rid])
    features1 += [blue_synergy, red_synergy, blue_vs_red, red_vs_blue]
    features1 += lane_counters

    lane_counters_flip = [0] * 5
    for role_id in range(5):
        red_champ = [c for c, r in zip(red_ids, red_roles) if r == role_id]
        blue_champ = [c for c, r in zip(blue_ids, blue_roles) if r == role_id]
        if red_champ and blue_champ:
            lane_counters_flip[role_id] = counter_score[red_champ[0]][blue_champ[0]]

    features2 = []
    for cid, rid in zip(red_ids + blue_ids, red_roles + blue_roles):
        features2.extend([cid, rid])
    features2 += [red_synergy, blue_synergy, red_vs_blue, blue_vs_red]
    features2 += lane_counters_flip

    return features1, features2

def estimate_champion_impact_batched(champion, role, n_samples=1000, neutral_champion="Garen"):
    role_idx = roles.index(role)
    ally_roles = roles[:role_idx] + roles[role_idx+1:]

    features = []

    for _ in range(n_samples):
        allies = random.sample([c for c in all_champions if c != champion and c != neutral_champion], 4)
        enemies = random.sample([c for c in all_champions if c != champion and c != neutral_champion], 5)

        blue_team = [(champion, role)] + list(zip(allies, ally_roles))
        neutral_team = [(neutral_champion, role)] + list(zip(allies, ally_roles))
        red_team = list(zip(enemies, roles))

        try:
            f1, f2 = build_feature_vectors(blue_team, red_team)
            f3, f4 = build_feature_vectors(neutral_team, red_team)
            features.extend([f1, f2, f3, f4])
        except:
            continue

    if not features:
        return 0.0

    preds = model.predict_proba(np.array(features))[:, 1]
    win_with = (preds[0::4] + (1 - preds[1::4])) / 2
    win_neutral = (preds[2::4] + (1 - preds[3::4])) / 2
    return np.mean(win_with - win_neutral)

def single_champion_impact(args):
    champion, role, n_samples, neutral_champion = args
    try:
        score = estimate_champion_impact_batched(champion, role, n_samples=n_samples, neutral_champion=neutral_champion)
        return role, champion, score
    except:
        return role, champion, 0.0

def build_champion_tierlist(df, threshold=33, n_samples=1000, neutral_champion="Garen"):
    pair_counts = Counter()
    for _, row in df.iterrows():
        pair_counts[(row['championName'], row['role'])] += 1

    tierlist = defaultdict(dict)
    tasks = [(champ, role, n_samples, neutral_champion) for (champ, role), count in pair_counts.items() if count >= threshold]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(single_champion_impact, args) for args in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating champions"):
            role, champ, score = future.result()
            tierlist[role][champ] = score

    return tierlist

def print_tierlist(tierlist, top_k=None):
    for role in roles:
        print(f"\n===== {role} TIER LIST =====")
        sorted_champs = sorted(tierlist[role].items(), key=lambda x: x[1], reverse=True)
        if top_k:
            sorted_champs = sorted_champs[:top_k]
        for i, (champ, val) in enumerate(sorted_champs):
            print(f"{i+1}. {champ:15s} ({val:+.4f})")

def save_tierlist_to_csv(tierlist, filepath="tierlist-test-15.13.csv"):
    rows = []
    for role, champs in tierlist.items():
        for champ, score in champs.items():
            rows.append({"Role": role, "Champion": champ, "ImpactScore": score})
    df = pd.DataFrame(rows)
    df.sort_values(by=["Role", "ImpactScore"], ascending=[True, False], inplace=True)
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    tierlist = build_champion_tierlist(champion_data, threshold=73, n_samples=50000)
    print_tierlist(tierlist)
    save_tierlist_to_csv(tierlist)
