"""The /view/* publication plates answer errors as JSON with a reason.

The Figures › Plates tab shows the server's own error text when a plate is
unavailable, so a bad format or a missing calibration must not come back as a
bare HTML abort page.
"""
from __future__ import annotations

import pytest

from euclid_polish.web.app import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


@pytest.mark.parametrize("path", [
    "/view/population-atlas",
    "/view/star-population-calibration",
    "/view/galaxy-distribution-plate",
])
def test_unknown_plate_format_is_a_json_400_naming_the_formats(client, path):
    response = client.get(f"{path}?format=bmp")
    assert response.status_code == 400
    assert response.is_json
    assert "png" in response.get_json()["error"]
