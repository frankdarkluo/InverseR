import pytest

from inverser.sources import Source, SourceAsset, require_acquisition


def source(acquire="allowed", process="allowed", access="ready"):
    asset = SourceAsset(
        asset_id="clip",
        url="https://upload.wikimedia.org/example.ogv",
        rights_page="https://commons.wikimedia.org/example",
        expected_bytes=100,
    )
    return Source(
        source_id="example",
        url=asset.rights_page,
        game="snooker",
        license="CC-BY-SA-4.0",
        license_evidence=asset.rights_page,
        credit="Example rights holder",
        acquire=acquire,
        process=process,
        redistribute="allowed",
        access=access,
        disposition="test fixture",
        assets=(asset,),
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"acquire": "unverified"}, {"process": "denied"}, {"access": "blocked"}],
)
def test_accessibility_does_not_override_rights(kwargs):
    record = source(**kwargs)
    with pytest.raises(PermissionError):
        require_acquisition(record, record.assets[0])


def test_unregistered_asset_is_blocked():
    record = source()
    changed = record.assets[0].model_copy(update={"url": "https://example.com/other"})
    with pytest.raises(PermissionError, match="manifest"):
        require_acquisition(record, changed)
