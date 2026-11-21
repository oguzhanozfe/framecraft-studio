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
