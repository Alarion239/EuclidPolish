# Population model and calibration assumptions

Synthetic fields are generated from two point-process counts. For a field of
area (A), the numbers of stars and galaxies are

\[
N_\star\sim\operatorname{Poisson}(A\lambda_\star),\qquad
N_g\sim\operatorname{Poisson}(A\lambda_g).
\]

The binned galaxy-count calibration uses the Poisson likelihood (reported as
Cash deviance), rather than least-squares errors on counts. This is the
appropriate likelihood for independent low-count bins ([Cash 1979](https://doi.org/10.1086/156922)).
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

## Galaxies

Galaxy calibration now starts from one analytical latent distribution shared
by COSMOS and Euclid. It is not a splice of bright Euclid rows and faint COSMOS
rows, and it does not select or inspect any TNG galaxy.

The latent intensity uses the classical luminosity--size construction of
[de Jong & Lacey (2000)](https://arxiv.org/abs/astro-ph/9910066): an evolving
[Schechter (1976)](https://doi.org/10.1086/154079) luminosity function multiplied
by a lognormal size distribution at fixed luminosity and redshift,

\[
\begin{aligned}
\Phi(M,z) &= 0.4\ln 10\,\phi_\star(z)
  10^{0.4[\alpha(z)+1][M_\star(z)-M]}
  \exp[-10^{0.4(M_\star(z)-M)}],\\
\log_{10}R_{e,\rm kpc}\mid M,z &\sim
  \mathcal N\!\left(\mu_R(M,z),\sigma_R(M)^2\right).
\end{aligned}
\]

Writing \(u=\log_{10}(1+z)\) and \(x=M+20\), the current smooth extension is

\[
\begin{aligned}
M_\star(z)&=M_{\star,0}+Q_1u+Q_2u^2,\\
\alpha(z)&=\alpha_0+\alpha_zu,\\
\phi_\star(z)&=\phi_{\star,0}(1+z)^{P_1+P_2u},\\
\mu_R(M,z)&=a+bx+cu+b_2x^2+b_zxu,\\
\sigma_R(M)&=\sigma_{R,0}\exp(s_Mx).
\end{aligned}
\]

These additions retain one evolving Schechter luminosity function and one
conditional lognormal size law; they introduce neither galaxy classes nor a
morphology mixture. COSMOS supplies
\((m_{\rm F814W},z,R_e)\), so it constrains the redshift and size evolution.
Here \(m_{\rm F814W}\) is the COSMOS2025 SourceXtractor++ single-Sérsic
total-model magnitude (`mag_model_hst-f814w`), not the separate bulge+disk
total magnitude stored for the morphology-ready subset.
The compact local COSMOS artifact does not contain rest-frame magnitudes or
per-object K-corrections. The current coordinate is therefore
\(M_{\rm eff}=m_{\rm F814W}-DM(z)\), and the fitted evolution of
\(M_\star\) absorbs the mean F814W K-correction. It must not be interpreted as
a rest-frame luminosity function. COSMOS2025 supplies the catalogue context
and measured quantities ([Shuntov et al. 2025](https://arxiv.org/abs/2506.03243)).
Observed size--luminosity and size--redshift trends motivate the smooth size
law ([Shen et al. 2003](https://arxiv.org/abs/astro-ph/0301527);
[Shibuya et al. 2015](https://arxiv.org/abs/1503.07481)).

Euclid observes the same latent population through a deliberately small
response model:

\[
\begin{aligned}
m_{\rm VIS,true}&=24+s_m(m_{\rm F814W}-24)+\Delta_m
             +\epsilon_{\rm int},\quad
             \epsilon_{\rm int}\sim\mathcal N(0,\sigma_{\rm int}^2),\\
F_{\rm VIS,MER}&=F(m_{\rm VIS,true})+\epsilon_F,\quad
             \epsilon_F\sim\mathcal N(0,\sigma_F^2),\\
r_{\rm E}&=\sqrt{(s_rR_e)^2+r_0^2},\\
P({\rm detected})&=\operatorname{logit}^{-1}\!\left[
 \frac{m_{50}-m_{\rm VIS}}{w}-\eta(\langle\mu\rangle_e-24)\right],\\
\langle\mu\rangle_e&=m_{\rm VIS}+2.5\log_{10}(2\pi r_{\rm E}^2).
\end{aligned}
\]

The normalization, affine F814W-to-VIS transfer, and its intrinsic scatter are
fitted first using only the high-S/N \(m_{\rm VIS}<24\) counts. They are then
frozen while the size response and faint completeness are fitted to the full
magnitude--size plane. The measurement term is not a second free scatter:
\(\sigma_F\) is the galaxy-probability-weighted median of
\(\mathtt{FLUXERR\_VIS\_APER}\) from the cached MER catalogue. The likelihood
integrates the intrinsic magnitude scatter and then adds Gaussian noise in
flux space, where the catalogue uncertainty is approximately homoscedastic.
This avoids the invalid faint-source approximation
\((2.5/\ln 10)\sigma_F/F_{\rm observed}\). The pre-observation TNG target uses
only \(\sigma_{\rm int}\), so the renderer and detector model do not receive
Euclid catalogue noise twice. The bright threshold is a project validation
choice, not a literature constant.

Here \(r_{\rm E}\) is the circularized MER detection proxy
`0.1 arcsec/pixel * SEMIMAJOR_AXIS * sqrt(1-ELLIPTICITY)`, not a fitted Euclid
half-light radius. Euclid galaxies contribute fractional count
\(1-\mathtt{POINT\_LIKE\_PROB}\). The surface-brightness term is an explicit
selection model, motivated by the general fact that detectability is joint in
flux and apparent size; its exact form, pivots and bounds are project choices.
The Euclid Q1 MER paper defines the relevant catalogue context
([Euclid Collaboration 2025](https://arxiv.org/abs/2503.15305)).

The mean Schechter parameters and Euclid response are estimated by the Poisson
likelihood in binned two-dimensional planes. After fitting the COSMOS mean, an
extra-Poisson fractional-scatter parameter \(\tau\) is estimated with
\(\operatorname{Var}(N)=\mu+(\tau\mu)^2\). It changes the uncertainty model,
not the fitted mean. Because COSMOS is one field and the model is imperfect,
\(\tau\) is count overdispersion rather than a measurement of pure cosmic
variance. The conditional size parameters are estimated in log space with
deterministic robust clipping and a heteroscedastic Gaussian likelihood. The local
standard errors reported by the interface come from the optimizer Jacobian;
they are curvature diagnostics, not full systematic uncertainties.

## Staged TNG draw target

The fitted COSMOS--Euclid cube is now used only for geometry and physical
conditioning. It is marginalized over its former brightness axis to draw
(R_e) first and then (z\mid R_e). PHZ activity, mass, and sSFR conditionals
are also marginalized over brightness before choosing a TNG donor. The Q1
MER+PHZ `FLUX_VIS_2FWHM_APER` law is independent:

\[
\log_{10}(dN/dA/dm_{2\mathrm{FWHM}})=a_gm_{2\mathrm{FWHM}}+b_g,
\qquad 14\leq m_{2\mathrm{FWHM}}<29.
\]

The straight region is the widest consecutive positive-count interval spanning
at least 4 mag with (R^2\geq0.998). The checked Q1 input ends at 28, so 28--29
is explicitly an extrapolation. The law's analytical 14--29 integral sets the
galaxy surface density.

After donor selection, redshift transport, and resizing to the requested
half-light radius, the generator draws the independent 2FWHM magnitude. It
draws a separate empirical VIS PSF member, circularizes its radial profile,
convolves the resized VIS stamp, and measures a circular aperture of radius one
FWHM (diameter (2\times\mathrm{FWHM})). One shared scalar then sets that
aperture flux and preserves all four TNG colours. The dirty-image forward model
still draws its own fresh PSF. Zero or non-finite aperture stamps are rejected
and resampled; total-flux fallback is not used.

The legacy geometry cube reconstructed for this staging is

\[
p_{\rm draw}(z,m_{\rm VIS,true},R_e)
=a_{\rm field}\int p_{\rm latent}(z,m_{\rm F814W},R_e)\,
\mathcal N(m_{\rm VIS,true}\mid\bar m_{\rm VIS},\sigma_{\rm int}^2)\,
d m_{\rm F814W}.
\]

It supplies geometry and conditional physical state, not the final brightness
or scene normalization. TNG supplies morphology; its native population
distribution is not the statistical target.

## Current scope and validation status

The version-3 activation combines a fingerprinted geometry model with the
versioned Q1 straight magnitude law. Older empirical-CDF and cubic artifacts
fail closed. The calibration interface plots the brightness-marginalized
geometry target beside the independent Q1 VIS 2FWHM straight brightness law.
It also retains observation-response diagnostics for survey apparent
magnitude, redshift, angular size, median size at fixed magnitude, derived mean
surface brightness, the fitted Euclid completeness surface, and
four-fold held-out validation across the twelve cached Euclid cones.
The fitting step itself neither reads a TNG catalogue nor renders an image.
After explicit WebUI activation, synthetic jobs embed a compact fingerprinted
copy of the fitted parameters, reconstruct this same three-dimensional draw
cube on each worker, and sample its cells. TNG atlas morphologies are assigned
independently with diversity balancing; this is an explicit randomness model,
not an inferred COSMOS--TNG morphology relation. The old empirical-row/TNG
population artifact is not accepted as this new fit.

A one-component Schechter × lognormal law is a scientific baseline, not a
guarantee of adequacy. A poor posterior-predictive comparison means the model
must be expanded, for example to separate star-forming and quiescent
components or a mixture model. Flexible forward-model approaches such as
[GalSBI](https://arxiv.org/abs/2412.08701) provide a relevant next-step
reference, but are not part of the current standard fit.

## What is fitted versus imposed

The galaxy luminosity function, conditional size relation, photometric
transfer, size response and surface-brightness-dependent completeness are
fitted. The analytical families, cosmology, fitting windows, binning, clipping,
response pivots and quality thresholds are imposed choices. Homogeneous
positions and Poisson scene counts remain simulation assumptions. No TNG
morphology distribution is fitted in this stage.
