"""Focused contracts for typed TNG donor and rendered-stamp values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.image import Image, Role
from euclid_polish.tng.image import TNGSurfaceBrightnessImage
from euclid_polish.tng.types import (
    NativePhotometry,
    NominalRadiusGeometry,
    PhysicalRedshiftGeometry,
    RenderedTNG,
    TNGRedshiftTransform,
    TNGRenderTrace,
    TNGRotation,
    TNGView,
    TotalVISNormalization,
    VIS2FWHMNormalization,
)

BANDS = tuple(Config.LR_INPUT_BAND_NAMES)


def _write_skirt_frame(path: Path, value: float, *, pixel_scale_pc: float = 100.0) -> None:
    data = np.full((8, 10), value, dtype=">f4")
    hdu = fits.PrimaryHDU(data)
    hdu.header["BUNIT"] = "MJy/sr"
    hdu.header["CRPIX1"] = 5.5
    hdu.header["CRVAL1"] = 0.0
    hdu.header["CDELT1"] = pixel_scale_pc
    hdu.header["CUNIT1"] = "pc"
    hdu.header["CRPIX2"] = 4.5
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CDELT2"] = pixel_scale_pc
    hdu.header["CUNIT2"] = "pc"
    hdu.writeto(path)


def _write_view(tmp_path: Path, *, padded: bool = False) -> TNGView:
    galaxy_dir = tmp_path / "7"
    galaxy_dir.mkdir()
    file_id = "000007" if padded else "7"
    for index, fits_band in enumerate(("VIS", "Y", "J", "H"), start=1):
        _write_skirt_frame(
            galaxy_dir / f"TNG{file_id}_O2_Euclid_{fits_band}.fits",
            float(index),
        )
    return TNGView(
        galaxy_dir=galaxy_dir,
        subhalo_id="7",
        orientation=2,
        native_re_px=20.0,
        radius_manifest_fingerprint="manifest-fixture",
    )


def _nominal_trace(*, target_re_arcsec: float = 0.5) -> TNGRenderTrace:
    view = TNGView(
        galaxy_dir=Path("unused"),
        subhalo_id="42",
        orientation=3,
        native_re_px=20.0,
        radius_manifest_fingerprint="manifest-fixture",
    )
    geometry = NominalRadiusGeometry(
        target_re_arcsec=target_re_arcsec,
        scale_factor=0.5,
        radius_rendering="euclid_sersic_shrink_only_v2",
        radius_renderer_fingerprint="r" * 64,
    )
    return TNGRenderTrace(
        view=view,
        rotation=TNGRotation(angle_deg=37.0),
        geometry=geometry,
        max_output_side=128,
    )


def _rendered(*, trace: TNGRenderTrace | None = None) -> RenderedTNG:
    base = np.ones((6, 8, 4), dtype=np.float32)
    base *= np.arange(1, 5, dtype=np.float32)[None, None, :]
    image = Image(
        data=base,
        pixel_scale_arcsec=0.05,
        band_names=BANDS,
        is_clean=True,
        role=Role.CLEAN,
    )
    return RenderedTNG(image=image, trace=trace or _nominal_trace())


def test_tng_view_resolves_padded_paths_and_loads_native_image(tmp_path):
    view = _write_view(tmp_path, padded=True)

    assert view.fits_path("VIS").name == "TNG000007_O2_Euclid_VIS.fits"
    assert len(view.file_identity()) == 4
    assert view.native_re_arcsec(0.05) == pytest.approx(1.0)
    assert view.can_render(1.0, 0.05)
    assert not view.can_render(1.001, 0.05)

    image = view.load_surface_brightness()
    assert isinstance(image, TNGSurfaceBrightnessImage)
    assert image.shape == (8, 10, 4)
    assert image.bands == BANDS
    assert image.pixel_scale_pc == pytest.approx(100.0)
    np.testing.assert_allclose(image.plane("H_E"), 4.0)


def test_tng_view_validates_identity_and_render_query(tmp_path):
    with pytest.raises(ValueError, match="1..5"):
        TNGView(tmp_path, "7", 0, 10.0)
    with pytest.raises(ValueError, match="native_re_px"):
        TNGView(tmp_path, "7", 1, float("nan"))

    view = TNGView(tmp_path, "7", 1, 10.0)
    with pytest.raises(ValueError, match="unknown TNG FITS band"):
        view.fits_path("F814W")
    with pytest.raises(ValueError, match="target_re_arcsec"):
        view.can_render(0.0, 0.05)


def test_rotation_represents_one_applied_transform():
    data = np.arange(3 * 5, dtype=np.float32).reshape(3, 5, 1)
    image = TNGSurfaceBrightnessImage(
        data=data,
        bands=("VIS",),
        pixel_scale_pc=100.0,
    )

    quarter = TNGRotation(quarter_turns=5)
    assert quarter.quarter_turns == 1
    np.testing.assert_array_equal(
        quarter.apply(image).data,
        np.rot90(image.data, 1),
    )

    arbitrary = TNGRotation(angle_deg=370.0)
    assert arbitrary.angle_deg == pytest.approx(10.0)
    assert arbitrary.record_fields()["arbitrary_rotation"] is True
    with pytest.raises(ValueError, match="cannot also carry"):
        TNGRotation(quarter_turns=1, angle_deg=10.0)


def test_geometry_and_redshift_records_reject_illegal_states():
    with pytest.raises(ValueError, match="shrink-only"):
        NominalRadiusGeometry(0.5, 1.01, "radius", "fingerprint")
    with pytest.raises(ValueError, match="at least one"):
        PhysicalRedshiftGeometry(0, 2.0, 1.2, 0.3)

    physical = PhysicalRedshiftGeometry(4, 3.7, 1.2, 0.3)
    view = TNGView(Path("unused"), "42", 1, 20.0)
    with pytest.raises(ValueError, match="requires a redshift"):
        TNGRenderTrace(view, TNGRotation(), physical)

    transform = TNGRedshiftTransform(
        redshift=0.8,
        band_factors=(0.2, 0.3, 0.4, 0.5),
        drift_mode="sed_interp",
        drift_epsilon=-0.1,
        dimming_factor=0.2,
    )
    trace = TNGRenderTrace(view, TNGRotation(), physical, redshift=transform)
    fields = trace.record_fields()
    assert fields["rebin_factor"] == 4
    assert fields["z"] == pytest.approx(0.8)
    assert fields["redshift_band_factors"] == transform.band_factors


def test_native_photometry_is_compact_read_only_and_band_addressable():
    profile = np.array([3.0, 2.0, 1.0])
    photometry = NativePhotometry(profile, (10.0, 20.0, 30.0, 40.0))

    profile[0] = 99.0
    assert photometry.vis_mean_profile_mjy_sr[0] == pytest.approx(3.0)
    assert not photometry.vis_mean_profile_mjy_sr.flags.writeable
    assert photometry.band_sum("J_E") == pytest.approx(30.0)
    assert "array" not in repr(photometry)
    assert photometry != NativePhotometry([3.0, 2.0, 1.0], (10, 20, 30, 40))


def test_rendered_tng_validates_semantics_and_exposes_derived_fluxes():
    stamp = _rendered()

    assert isinstance(stamp.image, Image)
    assert stamp.image.is_clean is True
    assert stamp.image.role is Role.CLEAN
    assert stamp.shape == (6, 8, 4)
    assert stamp.bands == BANDS
    assert stamp.pixel_scale_arcsec == pytest.approx(0.05)
    assert stamp.fluxes_e == pytest.approx((48.0, 96.0, 144.0, 192.0))
    assert stamp.flux_e_per_band["H_E"] == pytest.approx(192.0)
    assert not stamp.data.flags.writeable
    assert not stamp.as_array().flags.writeable
    assert stamp.as_array() is stamp.data
    assert stamp.as_array(copy=True).flags.writeable
    assert "array" not in repr(stamp)
    with pytest.raises(ValueError, match="band is required"):
        stamp.plane()

    dirty = Image(
        data=np.ones((2, 2, 4), np.float32),
        pixel_scale_arcsec=0.05,
        band_names=BANDS,
        is_clean=False,
        role=Role.LR,
    )
    with pytest.raises(ValueError, match="clean images with Role.CLEAN"):
        RenderedTNG(
            image=dirty,
            trace=_nominal_trace(target_re_arcsec=0.5),
        )

    wrong_role = Image(
        data=np.ones((2, 2, 4), np.float32),
        pixel_scale_arcsec=0.05,
        band_names=BANDS,
        is_clean=True,
        role=Role.HR,
    )
    with pytest.raises(ValueError, match="clean images with Role.CLEAN"):
        RenderedTNG(image=wrong_role, trace=_nominal_trace())

    negative = Image(
        data=-np.ones((2, 2, 4), np.float32),
        pixel_scale_arcsec=0.05,
        band_names=BANDS,
        is_clean=True,
        role=Role.CLEAN,
    )
    with pytest.raises(ValueError, match="non-negative"):
        RenderedTNG(image=negative, trace=_nominal_trace())

    wrong_dtype = Image(
        data=np.ones((2, 2, 4), np.float64),
        pixel_scale_arcsec=0.05,
        band_names=BANDS,
        is_clean=True,
        role=Role.CLEAN,
    )
    with pytest.raises(ValueError, match="float32"):
        RenderedTNG(image=wrong_dtype, trace=_nominal_trace())

    nonfinite = Image(
        data=np.full((2, 2, 4), np.nan, np.float32),
        pixel_scale_arcsec=0.05,
        band_names=BANDS,
        is_clean=True,
        role=Role.CLEAN,
    )
    with pytest.raises(ValueError, match="finite"):
        RenderedTNG(image=nonfinite, trace=_nominal_trace())

    wrong_bands = Image(
        data=np.ones((2, 2, 4), np.float32),
        pixel_scale_arcsec=0.05,
        band_names=("VIS", "Y", "J", "H"),
        is_clean=True,
        role=Role.CLEAN,
    )
    with pytest.raises(ValueError, match="bands must be"):
        RenderedTNG(image=wrong_bands, trace=_nominal_trace())

    with pytest.raises(ValueError, match="nominal radius is inconsistent"):
        _rendered(trace=_nominal_trace(target_re_arcsec=0.4))


def test_rendered_tng_owns_an_immutable_image_snapshot() -> None:
    source = Image(
        data=np.ones((6, 8, 4), dtype=np.float32),
        pixel_scale_arcsec=0.05,
        band_names=BANDS,
        is_clean=True,
        role=Role.CLEAN,
        metadata={"origin": "caller"},
    )
    stamp = RenderedTNG(image=source, trace=_nominal_trace())

    source.data[...] = 9.0
    source.role = Role.LR
    source.metadata["origin"] = "mutated"
    np.testing.assert_array_equal(stamp.data, 1.0)
    assert stamp.image.role is Role.CLEAN
    assert stamp.image.metadata == {"origin": "caller"}

    detached = stamp.image
    detached.role = Role.LR
    detached.metadata["origin"] = "detached"
    detached.data = np.zeros_like(detached.data)
    assert stamp.image.role is Role.CLEAN
    assert stamp.image.metadata == {"origin": "caller"}
    np.testing.assert_array_equal(stamp.data, 1.0)
    stamp.validate()

    with pytest.raises(ValueError, match="WRITEABLE"):
        stamp.data.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        stamp.data[0, 0, 0] = 0.0


def test_rendered_tng_scaling_and_total_vis_normalization():
    stamp = _rendered()

    doubled = stamp.scaled(2.0)
    assert doubled.fluxes_e == pytest.approx(tuple(2.0 * value for value in stamp.fluxes_e))

    normalized = stamp.normalised_to_total_vis(120.0)
    assert normalized.flux_e("VIS") == pytest.approx(120.0, rel=1e-6)
    assert isinstance(normalized.trace.normalization, TotalVISNormalization)
    ratios = np.asarray(normalized.fluxes_e) / normalized.flux_e("VIS")
    np.testing.assert_allclose(ratios, (1.0, 2.0, 3.0, 4.0), rtol=1e-6)

    fields = normalized.record_fields()
    assert fields["target_re_arcsec"] == pytest.approx(0.5)
    assert fields["radius_scale_factor"] == pytest.approx(0.5)
    assert fields["target_vis_flux_e"] == pytest.approx(120.0)
    assert fields["flux_e_per_band"]["VIS"] == pytest.approx(120.0)


def test_redshift_and_aperture_normalizations_are_typed_trace_updates():
    stamp = _rendered()
    transform = TNGRedshiftTransform(
        redshift=1.0,
        band_factors=(0.5, 1.0, 1.5, 2.0),
        drift_mode="parametric",
        drift_epsilon=0.2,
        dimming_factor=0.125,
    )
    transformed = stamp.transformed_at_redshift(transform)
    assert transformed.fluxes_e == pytest.approx((24.0, 96.0, 216.0, 384.0))
    assert transformed.trace.redshift is transform

    aperture = VIS2FWHMNormalization(
        target_flux_e=240.0,
        achieved_flux_e=240.0,
        brightness_scale=10.0,
        psf_fwhm_arcsec=0.16,
        psf_source="fixture-psf",
        psf_fingerprint="p" * 64,
    )
    normalized = transformed.normalised(aperture)
    fields = normalized.record_fields()
    assert normalized.trace.normalization is aperture
    assert fields["target_vis_2fwhm_flux_e"] == pytest.approx(240.0)
    assert np.isfinite(fields["target_vis_2fwhm_mag"])
    assert fields["mer_photometric_fwhm_arcsec"] == pytest.approx(0.16)
    assert fields["aperture_radius_arcsec"] == pytest.approx(0.16)
    assert fields["aperture_diameter_arcsec"] == pytest.approx(0.32)
    assert fields["aperture_psf_model"] == (
        "circular_gaussian_mer_photometric_fwhm"
    )
