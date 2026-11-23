import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from framecraft.puzzle_openings import GOAL, START, compare, draw_board, review_cut


class PuzzleOpeningTests(unittest.TestCase):
    def test_comparison_reproduces_committed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = compare(output)
            expected = Path(__file__).resolve().parents[1] / "docs/puzzle-openings/comparison.json"
            self.assertEqual(report, json.loads(expected.read_text(encoding="utf-8")))
            self.assertTrue(report["same_final_frame"])
            self.assertEqual(report["cuts"]["move-first"]["decision"], "meets_brief")
            late = report["cuts"]["setup-first"]
            self.assertEqual(late["decision"], "revise")
            self.assertEqual({key for key, passed in late["checks"].items() if not passed},
                             {"motion_in_time", "goal_in_time"})
            for name, evidence in report["cuts"].items():
                self.assertEqual(review_cut(expected.parent / f"{name}.gif"), evidence)

    def test_a_direct_jump_cannot_satisfy_the_route(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jump.gif"
            draw_board(START).save(path, save_all=True, append_images=[draw_board(GOAL)],
                                   duration=[200, 600], loop=0)
            cut = review_cut(path)
            self.assertTrue(cut["checks"]["goal_in_time"])
            self.assertFalse(cut["checks"]["passes_turn_before_goal"])
            self.assertEqual(cut["decision"], "revise")

    def test_an_erased_marker_is_not_movement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "erased.gif"
            first = draw_board(START)
            first.save(path, save_all=True, append_images=[Image.new("RGB", first.size)],
                       duration=[100, 500], loop=0)
            cut = review_cut(path)
            self.assertIsNone(cut["first_motion_ms"])
            self.assertFalse(cut["checks"]["single_marker_visible"])
            self.assertEqual(cut["decision"], "revise")
