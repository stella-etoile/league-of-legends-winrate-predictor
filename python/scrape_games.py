import requests
import pandas as pd
import time
from requests.exceptions import ChunkedEncodingError
from urllib3.exceptions import ProtocolError
from tqdm import tqdm

API_KEY = "RGAPI-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
REGION = "na1"
PUUID_REGION = "americas"

headers = {
    "X-Riot-Token": API_KEY
}

RANK = "MASTER"
PATCH_VER = "15.14"
PREV_PATCH = "15.13"
INPUT_FILE = RANK + "_PUUID.csv"
OUTPUT_FILE = RANK + "_MATCHSTATS-" + PATCH_VER + ".csv"
PREV_PATCH = "COMBINED_MATCHSTATS-" + PREV_PATCH + ".csv"

def get_count(rank):
    return 100

COUNT = get_count(RANK)

match_ids = set()
match_stats = []
try:
    existing_df = pd.read_csv(OUTPUT_FILE)
    match_ids = set(existing_df["match_id"].unique())
    match_stats = existing_df.to_dict("records")
    print(f"Loaded {len(match_ids)} existing match_ids")
except FileNotFoundError:
    match_ids = set()
    match_stats = []
    print("/No previous data found. Starting fresh.")

try:
    prev_df = pd.read_csv(PREV_PATCH)
    prev_match_ids = set(prev_df["match_id"].unique())
    print(f"Loaded {len(prev_match_ids)} previous patch match_ids")
except FileNotFoundError:
    prev_match_ids = set()
    print("/No previous patch data found. Continuing without it.")

df = pd.read_csv(INPUT_FILE)
puuid_list = df["puuid"].tolist()

def safe_request(url, params=None):
    for _ in range(3):
        try:
            res = requests.get(url, headers=headers, params=params)
            if res.status_code == 200:
                return res
            else:
                print(f"  Status code: {res.status_code}")
                return None
        except (ChunkedEncodingError, ProtocolError) as e:
            print(f"  Retrying due to connection error: {e}")
            time.sleep(2)
    return None

def get_recent_match_ids(puuid, count=COUNT):
    url = f"https://{PUUID_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"start": 0, "count": count}
    res = safe_request(url, params)
    if res:
        time.sleep(1.2)
        return res.json()
    return []

def get_match_data(match_id):
    url = f"https://{PUUID_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    res = safe_request(url)
    if res:
        time.sleep(1.2)
        return res.json()
    return None

start_time = time.time()

for i, puuid in enumerate(tqdm(puuid_list, desc="Fetching matches")):
    ids = get_recent_match_ids(puuid, COUNT)
    print(f"\n[{i+1}/{len(puuid_list)}] Fetching {len(ids)}/{COUNT} matches for {puuid[:8]}", flush=True)
    for match_id in ids:
        if match_id in match_ids:
            print(f"  Already saved: {match_id}", flush=True)
            continue
        elif match_id in prev_match_ids:
            print("  Skipped {match_id} in patch {PREV_PATCH}", flush=True)
            break
        data = get_match_data(match_id)
        if data and data["info"]["queueId"] == 420:
            game_version = data["info"]["gameVersion"]
            if game_version.startswith(PREV_PATCH):
                print(f"  Skipped {match_id} with gameVersion: {game_version} - moving to next player", flush=True)
                break
            match_ids.add(match_id)
            print(f"  Match: {match_id}", flush=True)
            for p in data["info"]["participants"]:
                match_stats.append({
                    "match_id": match_id,
                    "gameDuration": data["info"]["gameDuration"],
                    "gameVersion": game_version,
                    "puuid": p["puuid"],
                    "summonerName": p["summonerName"],
                    "championName": p["championName"],
                    "teamId": p["teamId"],
                    "win": p["win"],
                    "kills": p["kills"],
                    "deaths": p["deaths"],
                    "assists": p["assists"],
                    "goldEarned": p["goldEarned"],
                    "totalDamageDealtToChampions": p["totalDamageDealtToChampions"],
                    "visionScore": p["visionScore"],
                    "totalMinionsKilled": p["totalMinionsKilled"],
                    "champLevel": p["champLevel"],
                    "role": p["teamPosition"]
                })
    pd.DataFrame(match_stats).to_csv(OUTPUT_FILE, index=False)
    print(f"Total SoloQ matches saved: {int(len(match_stats)/10)}", flush=True)

elapsed = time.time() - start_time
print(f"\nDone. Total SoloQ matches saved: {int(len(match_stats)/10)}")
print(f"Elapsed time: {elapsed:.2f} seconds")