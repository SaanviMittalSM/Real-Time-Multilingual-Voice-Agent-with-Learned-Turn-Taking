"""Shared helpers for working with raw AMI corpus files."""

import xml.etree.ElementTree as ET
from pathlib import Path

NITE_NS = "{http://nite.sourceforge.net/}"


def speaker_channel_map(annotations_root: Path):
    """Returns {meeting_id: {nxt_agent_letter: headset_channel_index}}.

    The A/B/C/D speaker letters used in words.xml/dialog-act.xml filenames
    do NOT always map to Headset-0/1/2/3 in a fixed order - it varies per
    meeting, so this has to be read from meetings.xml rather than assumed.
    """
    tree = ET.parse(annotations_root / "corpusResources" / "meetings.xml")
    result = {}
    for meeting in tree.getroot():
        meeting_id = meeting.get("observation")
        if not meeting_id:
            continue
        speakers = {}
        for speaker in meeting:
            letter = speaker.get("nxt_agent")
            channel = speaker.get("channel")
            if letter is not None and channel is not None:
                speakers[letter] = int(channel)
        result[meeting_id] = speakers
    return result
