"""Download AMI Meeting Corpus audio (individual headset channels) for the
official "scenario-only" train/dev/test split.

Per-speaker headset channels are used (not the mixed track) because that's
the realistic analogue of a single user's microphone in deployment — the
turn detector has to work off one person's audio, not a studio mixdown.

Manual annotations (word timestamps, dialogue acts) are downloaded
separately — see the 22MB zip at
https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
"""

import http.client
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{meeting}.{channel}.wav"
CHANNELS = ["Headset-0", "Headset-1", "Headset-2", "Headset-3"]
SUB_LETTERS = "abcde"  # not every prefix has all of these; 404s are skipped

SPLITS = {
    "train": [
        "ES2002", "ES2005", "ES2006", "ES2007", "ES2008", "ES2009", "ES2010",
        "ES2012", "ES2013", "ES2015", "ES2016",
        "IS1000", "IS1001", "IS1002", "IS1003", "IS1004", "IS1005", "IS1006", "IS1007",
        "TS3005", "TS3008", "TS3009", "TS3010", "TS3011", "TS3012",
    ],
    "dev": ["ES2003", "ES2011", "IS1008", "TS3004", "TS3006"],
    "test": ["ES2004", "ES2014", "IS1009", "TS3003", "TS3007"],
}


def meeting_ids_for(prefixes):
    return [f"{prefix}{letter}" for prefix in prefixes for letter in SUB_LETTERS]


def download(url, dest: Path, max_retries=4):
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"

    last_error = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp_dest = dest.with_suffix(dest.suffix + ".part")
                with open(tmp_dest, "wb") as f:
                    f.write(resp.read())
                tmp_dest.replace(dest)  # atomic: never leaves a truncated file at the real path
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "missing"
            last_error = e
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.IncompleteRead) as e:
            last_error = e
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, ...

    print(f"  WARNING: giving up on {url} after {max_retries} attempts ({last_error})")
    return "failed"


def download_split(split_name, out_root: Path):
    prefixes = SPLITS[split_name]
    meetings = meeting_ids_for(prefixes)
    print(f"[{split_name}] trying {len(meetings)} candidate meeting IDs "
          f"({len(prefixes)} sessions x {len(SUB_LETTERS)} sub-letters)")

    found_meetings = []
    for meeting in meetings:
        got_any_channel = False
        for channel in CHANNELS:
            url = BASE_URL.format(meeting=meeting, channel=channel)
            dest = out_root / split_name / meeting / f"{meeting}.{channel}.wav"
            status = download(url, dest)
            if status in ("ok", "cached"):
                got_any_channel = True
            time.sleep(0.1)  # be polite to the server
        if got_any_channel:
            found_meetings.append(meeting)
            print(f"  {meeting}: downloaded")
        # meetings that don't exist (e.g. no 'e' sub-letter) 404 on all channels silently

    print(f"[{split_name}] {len(found_meetings)} real meetings found: {found_meetings}")
    return found_meetings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "dev", "test", "all"])
    args = parser.parse_args()

    out_root = config.DATA_DIR / "raw" / "ami_audio"
    splits_to_run = ["train", "dev", "test"] if args.split == "all" else [args.split]
    for split in splits_to_run:
        download_split(split, out_root)
