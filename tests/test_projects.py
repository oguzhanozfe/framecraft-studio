from concurrent.futures import ThreadPoolExecutor

import pytest
from framecraft.projects import archive, library, project_folder


def test_project_iterations_are_idempotent_and_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMECRAFT_HOME", str(tmp_path))
    one = tmp_path / "cut-1.mp4"
    two = tmp_path / "cut-2.mp4"
    one.write_bytes(b"one")
    two.write_bytes(b"two")
    with ThreadPoolExecutor(2) as pool:
        a, b = pool.map(lambda p: archive("ASUS", p), [one, one])
    assert a == b and a.name == "ASUS - UGC 001.mp4"
    assert archive("ASUS", two).name == "ASUS - UGC 002.mp4"
    assert archive("Puzzle", one).name == "Puzzle - UGC 001.mp4"
    assert [r["number"] for r in library()[0]["editions"]] == [2, 1]
    with pytest.raises(ValueError):
        project_folder("..")
