import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from tqdm import tqdm
import xgboost as xgb

synergy_score = np.load("synergy_score-15.13.npy")
counter_score = np.load("counter_score-15.13.npy")

df = pd.read_csv("COMBINED_MATCHSTATS-15.13.csv")
df = df.dropna(subset=["championName", "role", "win"])

roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
role_to_idx = {r: i for i, r in enumerate(roles)}
champion_to_idx = {name: i for i, name in enumerate(df['championName'].unique())}
idx_to_champion = {i: name for name, i in champion_to_idx.items()}
all_champions = list(champion_to_idx.keys())

model = xgb.XGBClassifier()
model.load_model("xgb_champion_model-15.13.json")

def cache_with_stats(maxsize=1000000):
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize)(func)
        hits = misses = 0
        @wraps(cached_func)
        def wrapper(*args):
            nonlocal hits, misses
            before = cached_func.cache_info().hits
            result = cached_func(*args)
            after = cached_func.cache_info().hits
            if after > before:
                hits += 1
            else:
                misses += 1
            if (hits + misses) % 300000 == 0:
                percentage = round(hits/(hits+misses)*100.0,1)
                print(f"[{func.__name__}] cache stats: {hits} hits, {misses} misses, percentage: {percentage}%")
            return result
        wrapper.cache_clear = cached_func.cache_clear
        wrapper.cache_info = cached_func.cache_info
        return wrapper
    return decorator

@cache_with_stats()
def cached_average_synergy(team_key):
    team = list(team_key)
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    return np.mean([synergy_score[team[i]][team[j]] for i, j in pairs])

def cached_average_counter(offense_key, defense_key):
    offense = list(offense_key)
    defense = list(defense_key)
    return np.mean([counter_score[o][d] for o in offense for d in defense])

def cached_lane_counters(blue_key, red_key):
    blue = list(blue_key)
    red = list(red_key)
    result = [0] * 5
    for role_id in range(5):
        blue_champ = [champion_to_idx[c] for c, r in blue if role_to_idx[r] == role_id]
        red_champ = [champion_to_idx[c] for c, r in red if role_to_idx[r] == role_id]
        if blue_champ and red_champ:
            result[role_id] = counter_score[blue_champ[0]][red_champ[0]]
    return tuple(result)

def predict_winrate_cached(blue_key, red_key):
    blue_team = list(blue_key)
    red_team = list(red_key)
    blue_ids = [champion_to_idx[c] for c, _ in blue_team]
    red_ids = [champion_to_idx[c] for c, _ in red_team]
    blue_roles = [role_to_idx[r] for _, r in blue_team]
    red_roles = [role_to_idx[r] for _, r in red_team]

    blue_synergy = cached_average_synergy(tuple(blue_ids))
    red_synergy = cached_average_synergy(tuple(red_ids))
    blue_vs_red = cached_average_counter(tuple(blue_ids), tuple(red_ids))
    red_vs_blue = cached_average_counter(tuple(red_ids), tuple(blue_ids))
    lane_counters = cached_lane_counters(blue_key, red_key)

    flat_feats = []
    for champ_id, role_id in zip(blue_ids + red_ids, blue_roles + red_roles):
        flat_feats.extend([champ_id, role_id])
    flat_feats += [blue_synergy, red_synergy, blue_vs_red, red_vs_blue]
    flat_feats += list(lane_counters)
    x_blue = np.array(flat_feats).reshape(1, -1)

    lane_counters_flip = cached_lane_counters(red_key, blue_key)
    flat_feats_flip = []
    for champ_id, role_id in zip(red_ids + blue_ids, red_roles + blue_roles):
        flat_feats_flip.extend([champ_id, role_id])
    flat_feats_flip += [red_synergy, blue_synergy, red_vs_blue, blue_vs_red]
    flat_feats_flip += list(lane_counters_flip)
    x_red = np.array(flat_feats_flip).reshape(1, -1)

    prob_blue = model.predict_proba(x_blue)[0][1]
    prob_red = model.predict_proba(x_red)[0][1]
    return prob_blue, prob_red

def estimate_champion_value(champion_name, role, n_samples=50, neutral_champion="Garen"):
    impacts = []
    role_index = roles.index(role)
    for _ in range(n_samples):
        allies = random.sample([c for c in all_champions if c != champion_name and c != neutral_champion], 4)
        enemies = random.sample([c for c in all_champions if c != champion_name and c != neutral_champion], 5)
        ally_roles = roles[:role_index] + roles[role_index+1:]
        blue_team = tuple([(champion_name, role)] + [(a, r) for a, r in zip(allies, ally_roles)])
        red_team = tuple((e, r) for e, r in zip(enemies, roles))
        try:
            base_win, _ = predict_winrate_cached(blue_team, red_team)
            neutral_team = tuple([(neutral_champion, role)] + [(a, r) for a, r in zip(allies, ally_roles)])
            neutral_win, _ = predict_winrate_cached(neutral_team, red_team)
            impacts.append(base_win - neutral_win)
        except:
            continue
    return np.mean(impacts) if impacts else 0.0

def single_champion_impact(args):
    champion, role, n_samples, neutral_champion = args
    try:
        impact = estimate_champion_value(champion, role, n_samples, neutral_champion)
        return role, champion, impact
    except:
        return role, champion, 0.0

def build_champion_tierlist_filtered(df, threshold=50, n_samples=50):
    pair_counts = Counter()
    for _, row in df.iterrows():
        pair_counts[(row['championName'], row['role'])] += 1

    tierlist = defaultdict(dict)
    tasks = [(champ, role, n_samples, "Garen") for (champ, role), count in pair_counts.items() if count >= threshold]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(single_champion_impact, args) for args in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating champions"):
            role, champion, impact = future.result()
            tierlist[role][champion] = impact

    for role in roles:
        print(f"Finished evaluating {role}")
    return tierlist

def print_tierlist(tierlist, roles, top_k=None):
    for role in roles:
        print(f"\n===== {role} TIER LIST =====")
        sorted_champs = sorted(tierlist[role].items(), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            sorted_champs = sorted_champs[:top_k]
        for i, (champ, val) in enumerate(sorted_champs):
            print(f"{i+1}. {champ:15s} ({val:+.4f})")

def save_tierlist_to_csv(tierlist, filepath="tierlist-15.13.csv"):
    rows = []
    for role, champs in tierlist.items():
        for champ, score in champs.items():
            rows.append({"Role": role, "Champion": champ, "ImpactScore": score})
    df = pd.DataFrame(rows)
    df.sort_values(by=["Role", "ImpactScore"], ascending=[True, False], inplace=True)
    df.to_csv(filepath, index=False)

tierlist = build_champion_tierlist_filtered(df, threshold=33, n_samples=10000)
print_tierlist(tierlist, roles)
save_tierlist_to_csv(tierlist, "tierlist-15.13.csv")
