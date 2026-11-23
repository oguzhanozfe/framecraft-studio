"""Render and review two edits of an original three-by-three puzzle using only Pillow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageSequence

START, TURN, GOAL = (48, 56), (48, 152), (144, 152)
AMBER = (255, 183, 62)
TICK_MS = 100
BRIEF = {"motion_by_ms": 500, "goal_by_ms": 1000, "hold_goal_ms": 300}
ROUTE = [START, (48, 88), (48, 120), TURN, (80, 152), (112, 152), GOAL]


def draw_board(token: tuple[int, int]) -> Image.Image:
    frame = Image.new("RGB", (192, 208), "#111d27")
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle((13, 21, 179, 187), radius=18, fill="#1f303c", outline="#546b72", width=2)
    for row in range(3):
        for col in range(3):
            x, y = 48 + col * 48, 56 + row * 48
            walkable = col == 0 or row == 2
            draw.rounded_rectangle((x-21, y-21, x+21, y+21), radius=8,
                                   fill="#31494e" if walkable else "#223540")
            if not walkable:
                for dx in (-6, 6):
                    draw.line((x+dx-2, y+3, x+dx+2, y-3), fill="#3c5660", width=2)
    draw.line([START, TURN, GOAL], fill="#8dc8af", width=3)
    x, y = GOAL
    draw.ellipse((x-15, y-15, x+15, y+15), fill="#31494e", outline="#b8ebd1", width=3)
    x, y = token
    draw.polygon([(x, y-12), (x+12, y), (x, y+12), (x-12, y)], fill="#493b32")
    draw.polygon([(x, y-9), (x+9, y), (x, y+9), (x-9, y)], fill=AMBER)
    draw.line((x-3, y-2, x, y-5), fill="#fff3bd", width=2)
    return frame


def write_cut(path: Path, first_move_tick: int) -> None:
    # Both edits last 2.2 seconds. Only the distribution of head and tail holds changes.
    positions = [ROUTE[max(0, min(tick - first_move_tick + 1, len(ROUTE) - 1))] for tick in range(22)]
    frames = [draw_board(position) for position in positions]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=TICK_MS, loop=0)


def read_marker(frame: Image.Image) -> tuple[int, int] | None:
    # The review is specific to this authored palette and marker size.
    # GIF disposal and repeated-frame consolidation are handled by the decoder.
    reference = Image.new("RGB", frame.size, AMBER)
    red, green, blue = ImageChops.difference(frame, reference).split()
    max_error = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    matches = max_error.point(lambda error: 255 if error == 0 else 0)
    bounds = matches.getbbox()
    if bounds is None or not 140 <= matches.histogram()[255] <= 185:
        return None
    left, top, right, bottom = bounds  # right and bottom are exclusive
    if right - left != 19 or bottom - top != 19:
        return None
    return (left + 9, top + 9)


def review_cut(path: Path) -> dict:
    elapsed = 0
    evidence = []
    with Image.open(path) as animation:
        for frame in ImageSequence.Iterator(animation):
            pixels = frame.convert("RGB")
            evidence.append((elapsed, read_marker(pixels)))
            elapsed += frame.info["duration"]
        final_frame_hash = hashlib.sha256(pixels.tobytes()).hexdigest()

    active = next((ms for ms, pos in evidence if pos is not None and pos != START), None)
    turn = next((ms for ms, pos in evidence if pos == TURN), None)
    goal = next((ms for ms, pos in evidence if pos == GOAL), None)
    terminal_hold = 0
    for ms, pos in reversed(evidence):
        if pos != GOAL:
            break
        terminal_hold = elapsed - ms
    checks = {
        "single_marker_visible": all(pos is not None for _, pos in evidence),
        "starts_at_source": evidence[0][1] == START,
        "passes_turn_before_goal": turn is not None and goal is not None and turn < goal,
        "motion_in_time": active is not None and active <= BRIEF["motion_by_ms"],
        "goal_in_time": goal is not None and goal <= BRIEF["goal_by_ms"],
        "holds_final_goal": terminal_hold >= BRIEF["hold_goal_ms"],
    }
    return {"decision": "meets_brief" if all(checks.values()) else "revise",
            "first_motion_ms": active, "first_goal_ms": goal, "final_goal_hold_ms": terminal_hold,
            "duration_ms": elapsed, "checks": checks, "final_frame_rgb_sha256": final_frame_hash}


def compare(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cuts = {}
    for name, start in [("move-first", 2), ("setup-first", 11)]:
        path = output / f"{name}.gif"
        write_cut(path, first_move_tick=start)
        cuts[name] = review_cut(path)
    report = {"source": "Original geometric puzzle, synthetic offline fixture.",
              "brief": BRIEF, "cuts": cuts,
              "same_final_frame": len({cut["final_frame_rgb_sha256"] for cut in cuts.values()}) == 1}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory for the GIFs and comparison.json")
    result = compare(parser.parse_args().output)
    for name, cut in result["cuts"].items():
        print(f"{name}: {cut['decision']} (motion {cut['first_motion_ms']} ms, goal {cut['first_goal_ms']} ms)")
