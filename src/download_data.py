"""Download the open data used in the experiments.

* UNOSAT building damage assessment of Mariupol (Livoberezhnyi and Zhovtnevyi districts; 14 March and
  7-12 May 2022), Humanitarian Data Exchange: https://data.humdata.org/dataset/89731ba0-f655-4b26-8d52-7167dbe84e08
* CrisisMMD v2.0 damage-severity splits (QCRI, CC BY-NC-SA 4.0), Hugging Face mirror QCRI/CrisisMMD.
  Images (~415 MB) are needed only to recompute the perception features (--images).

Usage: python src/download_data.py [--data-dir data/raw] [--images]
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import zipfile

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNOSAT_URL = ("https://unosat-maps.web.cern.ch/UA/CE20220223UKR/"
              "UNOSAT_LivoberezhnyiDistrict_DamageAssessment_12May2022_shp.zip")
HF_BASE = "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/main/"


def fetch(url, path, retries=5):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skip"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                return "ok"
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return f"http {r.status_code}"
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return "failed"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--images", action="store_true", help="also download the CrisisMMD images (~415 MB)")
    a = ap.parse_args()

    # UNOSAT shapefile
    udir = os.path.join(a.data_dir, "unosat")
    zpath = os.path.join(udir, "livoberezhnyi_shp.zip")
    print("UNOSAT:", fetch(UNOSAT_URL, zpath))
    with zipfile.ZipFile(zpath) as z:
        z.extractall(os.path.join(udir, "extracted"))

    # CrisisMMD damage-severity splits
    cdir = os.path.join(a.data_dir, "crisismmd")
    for split in ("train", "dev", "test"):
        print(f"CrisisMMD {split}:", fetch(HF_BASE + f"damage/{split}.json", os.path.join(cdir, f"damage_{split}.json")))

    if a.images:
        paths = []
        for split in ("train", "dev", "test"):
            with open(os.path.join(cdir, f"damage_{split}.json"), encoding="utf-8") as f:
                paths += [d["image_path"] for d in json.load(f)]
        paths = sorted(set(paths))
        print("CrisisMMD images:", len(paths))
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda p: fetch(HF_BASE + p, os.path.join(cdir, "images", p)), paths))
        print({k: res.count(k) for k in set(res)})


if __name__ == "__main__":
    main()
