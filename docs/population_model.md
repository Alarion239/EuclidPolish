# Population model and calibration assumptions

Synthetic fields are generated from two point-process counts. For a field of
area (A), the numbers of stars and galaxies are

\[
N_\star\sim\operatorname{Poisson}(A\lambda_\star),\qquad
N_g\sim\operatorname{Poisson}(A\lambda_g).
\]

Positions are a homogeneous Poisson process: this is an explicit simulation
assumption, not a fitted claim about galaxy clustering.

## Stars

The active stellar artifact uses one straight differential-count law,

\[
\log_{10}(dN/dA/dm)=a_\star m+b_\star,
\]

over VIS 12--25. It first requires `POINT_LIKE_PROB >= 0.9`; in each
0.1-mag VIS PSF-flux bin it reports both `SUM(POINT_LIKE_PROB)` as the expected
number of point sources without PHZ and `SUM(PHZ_STAR_PROB)` as the PHZ stellar
expectation. Both are divided by the released 63.1 deg² Q1 deep-field footprint
and the bin width. The raw number passing the point-source cut is retained
separately. The three deep fields are included; the separate
LDN 1641 commissioning field is excluded. No Gaia count enters this
normalization. Consecutive positive-count bins are searched automatically for
the widest interval spanning at least 2.5 mag with (R^2\geq0.99). Native Gaia
`G_AB` and Q1 PHZ VIS intervals are fitted with one shared slope and separate
intercepts; the Q1 intercept alone normalizes the final VIS law. Its analytical
12--25 integral sets the generator surface density.

The WebUI magnitude-density diagnostic additionally overlays the cached-cone
Gaia G distribution. It applies only the Gaia (E)DR3 Vega-to-AB zero-point
offset and does not transform Gaia G into VIS. Its surface density uses the
area of the field-centred Gaia cones. Gaia informs the shared count slope but
not the active Q1 normalization.

Gaia remains useful for a different quantity: matched Gaia--Euclid sources
constrain the temperature/colour locus. Gaia sampling uses twelve 0.35-degree
cones centred by spherical k-means on the cached Euclid sources with
`POINT_LIKE_PROB >= 0.9`; these geometric centres are not removed from the
query. The cones are retrieved synchronously from the ARI Gaia DR3 TAP mirror;
any TAP overflow fails closed instead of caching a truncated sample. Euclid flux errors use the fitted
heavy-tailed likelihood, and local `POINT_LIKE_PROB` weights the colour-locus
fit. The generator inverse-CDF samples the fitted finite-domain straight law
and then the magnitude-conditioned latent colour model for the four Euclid
bands. No empirical magnitude CDF, blackbody SED, polynomial colour map, or
legacy magnitude fitter is used for current generation. The photometric basis follows the Gaia DR3 synthetic
photometry work ([Gaia DR3 synthetic photometry](https://arxiv.org/abs/2206.06215));
the exact artifact version and cuts are project validation choices.

## Active Euclid-only galaxy model

The active galaxy model uses exactly two Euclid quantities:

\[
m=m_{\rm VIS,2FWHM},\qquad
r=\log_{10}(R_{e,\rm circ}/{\rm arcsec}),\qquad
R_{e,\rm circ}=R_{e,\rm major}\sqrt q.
\]

Q1 MER+PHZ aggregate counts set a three-part brightness density: measured
0.1-mag bins at the bright end, the fitted straight log-density law through
its break, and a flat imposed tail of 100 objects arcmin\(^{-2}\) mag\(^{-1}\)
through VIS 29.

The radius calibration is also a bounded aggregate, not an object catalogue.
It fits a broken Gaussian core in \(\log_{10}R_{e,\rm circ}\) plus a small
uniform-log-radius component. The broad component is used only through VIS
25.5; added fainter galaxies draw from the compact core.

The science-clean radius selection requires `VIS_DET = 1`, positive VIS
Sérsic flux, `POINT_LIKE_FLAG IS NULL`, `SPURIOUS_FLAG = 0`,
`DET_QUALITY_FLAG < 4`,
`PHZ_GAL_PROB >= 0.5`, `SERSIC_VISNIR_FLAGS = 0`,
\(0.05<q<1\), \(0.302<n<5.45\), and
\(R_{e,\rm major}<2a_{\rm detection}\). The last cut uses the documented
0.1-arcsec VIS pixel scale for `SEMIMAJOR_AXIS`. No lower
\(R_e>0.16\) cut is imposed, so unresolved faint galaxies remain available.
Every grouped bin is weighted by `PHZ_GAL_PROB`.

Generation marginalizes the two-dimensional law over brightness and draws
circularized VIS Sérsic \(R_e\) first. It resizes a random TNG donor to that radius, then
draws \(m\mid R_e\) from the same joint law. A separate empirical VIS PSF is
used to measure the resized stamp and one shared four-band factor matches the
drawn 2FWHM flux. COSMOS, detection-radius, and Kron-radius plots are diagnostic
overlays only.

For a donor with native VIS half-light radius \(R_{e,\rm native}\) pixels on
the 0.05-arcsec grid, the initial spatial resize is

\[
s_0=\frac{R_{e,\rm requested}}
          {0.05\,R_{e,\rm native}}.
\]

The renderer remeasures the resized stamp and iterates this factor until its
achieved half-light radius matches the request. Spatial resizing does not set
brightness. After resizing, the independent common four-band flux factor is
\(c=F_{\rm goal,2FWHM}/F_{\rm measured,2FWHM}\).

## Current scope and validation status

Version-10 activation contains only the empirical/fitted/flat brightness law
and the cleaned circularized-Sérsic-radius law. Versions 7--9 remain loadable
for reproducibility, but mixed old/new contracts fail closed. The fitting step
reads no TNG image; TNG is used only after activation as a random morphology
donor. This deliberately modest model aims for plausible source counts and
sizes rather than an exact catalogue replica.

## What is fitted versus imposed

The middle Q1 brightness coefficients, broken radius-core coefficients,
scatter, and broad-component fraction are fitted. The empirical bright bins,
finite 14--29 magnitude range, flat faint count density, compact-only
faint-radius policy, radius bounds, homogeneous positions, Poisson scene
counts, and random TNG donor assignment are imposed choices. No COSMOS or TNG
distribution is fitted.
