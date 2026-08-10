"""Derive turn-taking labels from AMI's NXT XML annotations.

AMI has no dedicated "turn" annotation layer, so turn boundaries are derived
from per-speaker word-level timestamps (forced-aligned), the way prior
turn-taking work on meeting corpora does it (see Roddy/Skantze/Harte,
Interspeech 2018 / ICMI 2018 / SIGDIAL 2019 for the general approach this
follows). Backchannel labels ARE directly annotated (dialogue-act type
"bck") and are used as ground truth rather than inferred.

Pipeline per meeting:
  1. Parse each speaker's words.xml -> ordered list of (word, start, end).
  2. Parse each speaker's dialog-act.xml -> word-id ranges tagged "bck".
  3. Group each speaker's words into utterances (consecutive words with a
     gap under UTTERANCE_GAP_S are the same utterance).
  4. Merge all speakers' utterances into one meeting-level timeline.
  5. For each utterance, label what happens at its end:
       - "shift"       : a different speaker's next utterance is a real
                         (non-backchannel) turn -> the floor changed hands.
       - "hold_short"  : same speaker resumes after a pause < HOLD_SPLIT_MS
                         (ordinary within-utterance-ish pause).
       - "hold_long"   : same speaker resumes after a pause >= HOLD_SPLIT_MS
                         ("needs more wait time" — a hesitation/thinking
                         pause that doesn't hand off the floor).
       - "backchannel" : the utterance itself is DA-tagged "bck" (e.g.
                         "mm-hmm", "yeah") - a class in its own right, not
                         a real turn attempt. When looking for what happens
                         after some OTHER utterance's pause, backchannels
                         are looked through rather than counted as the next
                         turn-taking event.

HOLD_SPLIT_MS is deliberately a tunable parameter, not a fixed constant —
it's the same knob the fixed-VAD-threshold baseline uses, so sweeping it is
part of the Phase 3 experiment, not a hardcoded assumption.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

NITE_NS = "{http://nite.sourceforge.net/}"
UTTERANCE_GAP_S = 0.1   # gap within which consecutive words are one utterance
HOLD_SPLIT_MS = 700     # same-speaker pause >= this -> "hold_long" not "hold_short"

# UTTERANCE_GAP_S must stay below the smallest fixed-VAD threshold evaluation.turn_eval
# tests (300ms) - otherwise no "hold" pause can ever be shorter than the merge gap itself,
# which silently caps the minimum observable pause and degenerates the 300ms comparison.

WORD_ID_RE = re.compile(r"id\(([^)]+)\)")


def parse_words(words_xml_path: Path):
    """Returns ordered list of dicts: {id, text, start, end, punc}."""
    tree = ET.parse(words_xml_path)
    words = []
    for w in tree.getroot():
        tag = w.tag.replace(NITE_NS, "")
        if tag != "w":
            continue  # skip <vocalsound>, <disfmarker>, etc. for now
        wid = w.get(f"{NITE_NS}id")
        start = w.get("starttime")
        end = w.get("endtime")
        if start is None or end is None:
            continue
        words.append({
            "id": wid,
            "text": w.text or "",
            "start": float(start),
            "end": float(end),
            "punc": w.get("punc") == "true",
        })
    return words


def parse_backchannel_word_ids(dialog_act_xml_path: Path, da_types_by_id):
    """Returns a set of word IDs that fall inside a 'bck' (Backchannel) DA span."""
    tree = ET.parse(dialog_act_xml_path)
    bck_ids = set()
    for dact in tree.getroot():
        da_type_id = None
        child_href = None
        for child in dact:
            tag = child.tag.replace(NITE_NS, "")
            if tag == "pointer" and child.get("role") == "da-aspect":
                href = child.get("href", "")
                m = WORD_ID_RE.search(href)
                if m:
                    da_type_id = m.group(1)
            if tag == "child":
                child_href = child.get("href", "")
        if da_types_by_id.get(da_type_id) != "bck":
            continue
        ids = WORD_ID_RE.findall(child_href or "")
        if len(ids) == 2:
            bck_ids.update(word_id_range(ids[0], ids[1]))
        elif len(ids) == 1:
            bck_ids.add(ids[0])
    return bck_ids


def word_id_range(start_id, end_id):
    """Expands 'meeting.speaker.words12'..'meeting.speaker.words18' to the full list."""
    prefix = re.match(r"(.+words)(\d+)$", start_id).group(1)
    start_n = int(re.match(r".+words(\d+)$", start_id).group(1))
    end_n = int(re.match(r".+words(\d+)$", end_id).group(1))
    return [f"{prefix}{n}" for n in range(start_n, end_n + 1)]


def parse_da_types(da_types_xml_path: Path):
    """Returns {nite_id: short_name} e.g. {'ami_da_1': 'bck'}."""
    tree = ET.parse(da_types_xml_path)
    out = {}

    def walk(elem):
        nid = elem.get(f"{NITE_NS}id")
        name = elem.get("name")
        if nid and name:
            out[nid] = name
        for child in elem:
            walk(child)

    walk(tree.getroot())
    return out


def words_to_utterances(words, backchannel_ids):
    """Groups consecutive words (small gaps) into utterances."""
    utterances = []
    current = []
    for w in words:
        if w["punc"]:
            continue
        if current and (w["start"] - current[-1]["end"]) > UTTERANCE_GAP_S:
            utterances.append(current)
            current = []
        current.append(w)
    if current:
        utterances.append(current)

    result = []
    for u in utterances:
        text = " ".join(w["text"] for w in u)
        is_backchannel = all(w["id"] in backchannel_ids for w in u) if backchannel_ids else False
        result.append({
            "text": text,
            "start": u[0]["start"],
            "end": u[-1]["end"],
            "is_backchannel": is_backchannel,
        })
    return result


def build_meeting_labels(meeting_id, annotations_root: Path, channel_map):
    words_dir = annotations_root / "words"
    da_dir = annotations_root / "dialogueActs"
    da_types = parse_da_types(annotations_root / "ontologies" / "da-types.xml")
    speaker_channels = channel_map.get(meeting_id, {})

    speaker_files = sorted(words_dir.glob(f"{meeting_id}.*.words.xml"))
    if not speaker_files:
        return None

    all_utterances = []  # each: {speaker, text, start, end, is_backchannel}
    for wf in speaker_files:
        speaker = wf.stem.split(".")[1]
        words = parse_words(wf)
        da_file = da_dir / f"{meeting_id}.{speaker}.dialog-act.xml"
        backchannel_ids = parse_backchannel_word_ids(da_file, da_types) if da_file.exists() else set()
        for u in words_to_utterances(words, backchannel_ids):
            u["speaker"] = speaker
            all_utterances.append(u)

    if not all_utterances:
        return None

    all_utterances.sort(key=lambda u: u["start"])

    labeled = []
    for i, u in enumerate(all_utterances):
        if u["is_backchannel"]:
            # The utterance itself IS a backchannel ("mm-hmm", "yeah") - this is one of the
            # four target classes directly, not a property of someone else's turn.
            labeled.append({
                "meeting_id": meeting_id,
                "speaker": u["speaker"],
                "headset_channel": speaker_channels.get(u["speaker"]),
                "text": u["text"],
                "utterance_start": u["start"],
                "utterance_end": u["end"],
                "pause_after_s": None,
                "next_speaker": None,
                "label": "backchannel",
            })
            continue

        # Find the next REAL (non-backchannel) utterance from any speaker. Backchannels
        # occurring during this speaker's pause don't count as "someone took the floor" -
        # they're looked through, not treated as the next turn-taking event.
        next_u, next_gap = None, None
        for cand in all_utterances[i + 1:]:
            if cand["start"] < u["end"] or cand["is_backchannel"]:
                continue
            next_u = cand
            next_gap = cand["start"] - u["end"]
            break

        if next_u is None:
            label = "shift"  # last real utterance of the meeting for this speaker; treat as turn end
        elif next_u["speaker"] == u["speaker"]:
            label = "hold_long" if next_gap * 1000 >= HOLD_SPLIT_MS else "hold_short"
        else:
            label = "shift"

        labeled.append({
            "meeting_id": meeting_id,
            "speaker": u["speaker"],
            "headset_channel": speaker_channels.get(u["speaker"]),
            "text": u["text"],
            "utterance_start": u["start"],
            "utterance_end": u["end"],
            "pause_after_s": next_gap,
            "next_speaker": next_u["speaker"] if next_u else None,
            "label": label,
        })

    return labeled


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "dev", "test"])
    args = parser.parse_args()

    from download_ami import SPLITS, meeting_ids_for  # noqa: E402
    from ami_utils import speaker_channel_map  # noqa: E402

    annotations_root = config.DATA_DIR / "raw" / "ami_annotations" / "extracted"
    audio_root = config.DATA_DIR / "raw" / "ami_audio" / args.split
    channel_map = speaker_channel_map(annotations_root)

    candidate_meetings = meeting_ids_for(SPLITS[args.split])
    available_meetings = [m.name for m in audio_root.iterdir()] if audio_root.exists() else []

    all_labels = []
    skipped_no_da, processed = [], []
    for meeting_id in candidate_meetings:
        if meeting_id not in available_meetings:
            continue
        labels = build_meeting_labels(meeting_id, annotations_root, channel_map)
        if labels is None:
            skipped_no_da.append(meeting_id)
            continue
        all_labels.extend(labels)
        processed.append(meeting_id)

    out_path = config.DATA_DIR / "manifests" / f"turn_labels_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_labels, f, indent=2)

    label_counts = {}
    for rec in all_labels:
        label_counts[rec["label"]] = label_counts.get(rec["label"], 0) + 1

    print(f"Processed {len(processed)} meetings, skipped {len(skipped_no_da)} (no annotations)")
    print(f"Skipped: {skipped_no_da}")
    print(f"Total labeled utterances: {len(all_labels)}")
    print(f"Label distribution: {label_counts}")
    print(f"Wrote {out_path}")
