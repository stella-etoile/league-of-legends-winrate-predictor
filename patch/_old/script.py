import requests
import pandas as pd
import time
from multiprocessing import Pool

# AGAINST TOS BUT IN THEORY, SHOULD INCREASE API CALLS

API_KEYS = [
    "RGAPI-863599a0-bcbf-4013-a4e9-6ede4f5eae1f",
    "RGAPI-57c934a7-849f-4828-b0c1-77ccc902605a",
    "RGAPI-2388eca1-f2fe-4caf-baca-99b70190d1f4",
    "RGAPI-5f802611-fc6e-4893-bc3a-6c6772467bef",
    "RGAPI-7be679df-9b76-4645-a8e7-764889a5cb57"
    # add more if you have them
]

PUUID_REGION = "americas"
MATCH_COUNT = 20

def get_recent_match_ids(puuid, api_key):
    url = f"https://{PUUID_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    headers = {"X-Riot-Token": api_key}
    res = requests.get(url, headers=headers, params={"start": 0, "count": MATCH_COUNT})
    print(f"Status: {res.status_code}, Response: {res.text}")
    if res.status_code != 200:
        return []
    return res.json()

def get_match_data(match_id, api_key):
    url = f"https://{PUUID_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": api_key}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        return None
    return res.json()

def worker(args):
    puuid_batch, api_key, worker_id = args
    match_ids_seen = set()
    rows = []

    for i, puuid in enumerate(puuid_batch):
        print(f"[Worker {worker_id}] {i+1}/{len(puuid_batch)}: {puuid[:8]}...", flush=True)
        try:
            match_ids = get_recent_match_ids(puuid, api_key)
            print(f"  -> Found {len(match_ids)} matches for {puuid[:8]}", flush=True)
            for match_id in match_ids:
                print(f"    Checking match {match_id}", flush=True)
                if match_id in match_ids_seen:
                    continue
                data = get_match_data(match_id, api_key)
                if not data or data["info"]["queueId"] != 420:
                    print(f"Status: {res.status_code}, Response: {res.text}")
                    continue
                match_ids_seen.add(match_id)
                for p in data["info"]["participants"]:
                    rows.append({
                        "match_id": match_id,
                        "puuid": p["puuid"],
                        "champion": p["championName"],
                        "win": p["win"],
                        "kills": p["kills"],
                        "deaths": p["deaths"],
                        "assists": p["assists"],
                        "goldEarned": p["goldEarned"],
                        "visionScore": p["visionScore"],
                        "cs": p["totalMinionsKilled"],
                        "role": p["teamPosition"],
                        "gameDuration": data["info"]["gameDuration"],
                        "gameVersion": data["info"]["gameVersion"]
                    })
                time.sleep(0.7)  # within safe rate
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
    return rows

if __name__ == "__main__":
    df = pd.read_csv("masters_puuids_part1.csv")
    puuid_list = df["puuid"].tolist()
    num_keys = len(API_KEYS)

    # Split PUUIDs across keys
    puuid_chunks = [puuid_list[i::num_keys] for i in range(num_keys)]
    args = [(chunk, API_KEYS[i], i) for i, chunk in enumerate(puuid_chunks)]

    # Start multiprocessing
    with Pool(processes=num_keys) as pool:
        results = pool.map(worker, args)

    # Combine results and save
    all_rows = [row for sublist in results for row in sublist]
    pd.DataFrame(all_rows).to_csv("multiprocessed_match_stats.csv", index=False)
    print(f"\nFinished. Saved {len(all_rows)} participant rows.")
