import os
import json
from pathlib import Path
from tqdm import tqdm

# === configuration ===
DATASET_DIR = "/home/anjitha/gaze/Interview/MultiModalDataset/MultiModalDataset"
SAVE_PATH = "/home/anjitha/gaze/Interview/processed_user_dataset.json"
MAX_USERS_PER_CLASS = 125
MAX_TOTAL_TWEETS = 50
REQUIRED_BOTH = 20
REQUIRED_TEXT_ONLY = 30

def is_valid_image(path):
    return path.exists() and path.stat().st_size > 0

def process_user(user_path):
    timeline_path = user_path / "timeline.txt"
    if not timeline_path.exists():
        return []

    posts_both, posts_text_only = [], []

    with open(timeline_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                tweet = json.loads(line.strip())
                tweet_id = str(tweet["id"])
                text = tweet.get("text", "").strip()
                img_path = user_path / f"{tweet_id}.jpg"
                has_image = is_valid_image(img_path)
                has_text = len(text) > 0

                post = {
                    "tweet_id": tweet_id,
                    "text": text if has_text else None,
                    "image_path": str(img_path) if has_image else None,
                    "has_text": has_text,
                    "has_image": has_image,
                    "timestamp": tweet.get("created_at", None)  # Optional
                }

                if has_text and has_image:
                    posts_both.append(post)
                elif has_text:
                    posts_text_only.append(post)
            except:
                continue

    # Take 20 with both, and 30 text-only
    if len(posts_both) >= REQUIRED_BOTH and len(posts_text_only) >= REQUIRED_TEXT_ONLY:
        selected = posts_both[:REQUIRED_BOTH] + posts_text_only[:REQUIRED_TEXT_ONLY]
        return selected
    else:
        return []

def filtered_dataset(root_dir):
    root = Path(root_dir)
    final_data = []

    for label_name, label in [("positive", 1), ("negative", 0)]:
        user_dir = root / label_name
        users = sorted(os.listdir(user_dir))
        count = 0

        for user_id in tqdm(users, desc=f"Processing {label_name}"):
            if count >= MAX_USERS_PER_CLASS:
                break

            user_path = user_dir / user_id
            if not user_path.is_dir():
                continue

            posts = process_user(user_path)
            if posts and len(posts) == MAX_TOTAL_TWEETS:
                final_data.append({
                    "user_id": user_id,
                    "label": label,
                    "posts": posts
                })
                count += 1

    return final_data

if __name__ == "__main__":
    data = filtered_dataset(DATASET_DIR)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} users to {SAVE_PATH}")
