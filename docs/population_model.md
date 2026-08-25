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
bands. The photometric basis follows the Gaia DR3 synthetic photometry work
([Gaia DR3 synthetic photometry](https://arxiv.org/abs/2206.06215));
the exact artifact version and cuts are project validation choices.

## Active Euclid-only galaxy model

The active galaxy model uses exactly two Euclid quantities:

\[
m=m_{\rm VIS,2FWHM},\qquad
r=\log_{10}(R_{e,\rm circ}/{\rm arcsec}),\qquad
R_{e,\rm circ}=R_{e,\rm major}\sqrt q.
\]

Q1 MER+PHZ aggregate counts set a continuous brightness density with a
three-segment bright bridge, the well-constrained main log-linear count law,
and a flat faint cap. The bridge ends at the fixed VIS joins
\(j_1=16.4\), \(j_2=19.0\), and \(j_3=20.9\). The main law then reaches an
imposed density of 100 objects arcmin\(^{-2}\) mag\(^{-1}\), which is held
constant through VIS 29. In log-density coordinates,

\[
\ell(m)=
\begin{cases}
a_1m+b_1, & 14\leq m<j_1,\\
a_2m+b_2, & j_1\leq m<j_2,\\
a_3m+b_3, & j_2\leq m<j_3,\\
a_m m+b_m, & j_3\leq m<m_{\rm flat},\\
\log_{10}(100), & m\geq m_{\rm flat}.
\end{cases}
\]

The three bridge slopes \(a_1,a_2,a_3\) are fitted to the bright 0.1-mag
brackets with bin-integrated Poisson deviance, including zero-count bins. The
three joins are fixed, not fitted. Continuity with the main line determines
all bridge intercepts recursively,

\[
b_3=(a_m-a_3)j_3+b_m,\qquad
b_2=(a_3-a_2)j_2+b_3,\qquad
b_1=(a_2-a_1)j_1+b_2.
\]

Thus the bright bridge contributes only three fitted degrees of freedom: its
three slopes.

The radius calibration is also a bounded aggregate, not an object catalogue.
It uses one straight conditional mean and one scatter at every magnitude,

\[
r\mid m\sim
\mathcal N\!\left(\mu_{23}+s(m-23),\,\sigma_r^2\right)
\quad\hbox{truncated to}\quad
\log_{10}(0.03)\leq r\leq\log_{10}(10).
\]

There is no bright-radius plateau, magnitude break, or separate broad
log-radius tail. Magnitude brackets with enough clean measurements enter the
fit through VIS 25.5, with each bracket capped in effective weight so the
millions of faint measurements do not erase the bright relation. The fitted
straight law itself applies across the complete VIS 14--29 generation range.

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
circularized VIS Sérsic \(R_e\) first. It chooses a diversity-balanced TNG donor
only from galaxies with at least one natively large-enough orientation, then
area-downsamples an eligible orientation to that radius. It never enlarges a
TNG stamp. The generator then draws \(m\mid R_e\) from the same joint law. The
normalized WebUI radius diagnostics keep two model marginals separate: one
weights the conditional radius law by the clean Q1 magnitude brackets, while
the other uses the complete generation law including its flat faint extension.
A separate empirical VIS PSF is used to measure the resized stamp and one
shared four-band factor matches the drawn 2FWHM flux. COSMOS,
detection-radius, and Kron-radius plots are diagnostic overlays only.

The image boundary is explicit throughout this path: `TNGView` identifies one
atlas orientation without loading its pixels, `TNGRenderer` owns the caches and
scientific transforms, and `RenderedTNG` pairs a read-only electron
`ImageCube` with typed render provenance. Native SKIRT cubes remain MJy/sr on a
physical pc/pixel grid until the renderer converts them to the Euclid angular
grid.

For a donor with native VIS half-light radius \(R_{e,\rm native}\) pixels on
the 0.05-arcsec grid, the spatial downsampling factor is

\[
s_0=\frac{R_{e,\rm requested}}
          {0.05\,R_{e,\rm native}} \leq 1.
\]

The renderer applies this nominal shrink-only scale from the donor's
pre-measured native radius; it does not remeasure the final rendered half-light
radius. If a sampled radius exceeds the available donor support, generation
rejects and redraws that geometry within a bounded retry instead of
interpolating or clamping it. Finite render support can still clip a large
native footprint, and the source catalog
records `render_support_clipped` so bounded image support remains explicit.
The requested radius is a model parameter, not a post-render measurement.
Spatial downsampling does not set brightness. After downsampling, the
independent common four-band flux factor is
\(c=F_{\rm goal,2FWHM}/F_{\rm measured,2FWHM}\).

## Current scope and validation status

Version 11 contains the continuous three-slope bright bridge/main/flat
brightness law and the single straight, no-tail circularized-Sérsic-radius
law. It is the only supported population artifact contract; older versions
must be refitted before use. The fitting step reads no TNG image; TNG is used only
after activation as a random morphology donor. This deliberately modest model
aims for plausible source counts and sizes rather than an exact catalogue
replica.

## What is fitted versus imposed

The main Q1 brightness coefficients, three bright-bridge slopes, straight
radius intercept and slope, and radius scatter are fitted. The fixed VIS joins
16.4, 19.0, and 20.9, finite 14--29 magnitude range, flat faint count density,
radius bounds, homogeneous positions, Poisson scene counts, and random TNG
donor assignment are imposed choices. No COSMOS or TNG distribution is
fitted.
