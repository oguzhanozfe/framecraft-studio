import pytest
from pydantic import ValidationError
from framecraft.product_link import ProductLink, listing


@pytest.mark.parametrize(
    ("url", "identity"),
    [
        (
            "https://www.amazon.com.tr/ASUS-Dual/dp/B0F8KJKC9Y?tag=example",
            ("product", "B0F8KJKC9Y"),
        ),
        (
            "https://apps.apple.com/tr/app/monument-valley/id728293409",
            ("game", "728293409"),
        ),
    ],
)
def test_links_reduce_to_a_store_id(url, identity):
    assert listing(ProductLink(url=url).url) == identity


def test_link_must_be_a_store_listing():
    with pytest.raises(ValidationError):
        ProductLink(url="https://example.com/dp/B0F8KJKC9Y")


def test_resume_campaign_preserves_rendered_takes(tmp_path, monkeypatch):
    from framecraft import server
    from framecraft.core import create_job, jobdir, record

    monkeypatch.setenv("FRAMECRAFT_HOME", str(tmp_path))
    queued = []
    monkeypatch.setattr(server.pool, "submit", lambda fn, job: queued.append(job))
    job = create_job("campaign")
    take = jobdir(job) / "existing.mp4"
    take.write_bytes(b"completed take")
    record(job, "failed", "intermittent weight read")
    assert server.retry(job) == {"id": job}
    with pytest.raises(ValueError):
        server.retry(job)
    assert queued == [job] and take.read_bytes() == b"completed take"
