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

The active Gaia+Euclid-v3 artifact supplies an empirical VIS-magnitude CDF and
a magnitude-conditioned latent stellar-locus mixture. The generator samples
the CDF and then the latent colour model for the four Euclid bands. Euclid
flux errors use the fitted heavy-tailed likelihood; `POINT_LIKE_PROB` is used
as a fractional membership weight when fitting the locus. No exponential
magnitude law, blackbody SED, polynomial colour map, or legacy fitter is used
for current generation. The photometric basis follows the Gaia DR3 synthetic
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

## Pre-observation TNG draw target

The distribution used to condition TNG rendering is

\[
p_{\rm draw}(z,m_{\rm VIS,true},R_e)
=a_{\rm field}\int p_{\rm latent}(z,m_{\rm F814W},R_e)\,
\mathcal N(m_{\rm VIS,true}\mid\bar m_{\rm VIS},\sigma_{\rm int}^2)\,
d m_{\rm F814W}.
\]

It includes the fitted Euclid field normalization and intrinsic band-transfer
scatter. It excludes the MER size floor, measurement error, catalogue
completeness and radius censoring. Those are observation effects applied only
after rendering. TNG supplies morphology; its native population distribution
is not the statistical target.

## Current scope and validation status

The version-2 artifact now records and plots the pre-observation TNG draw
target. It also produces comparisons of
apparent magnitude, redshift, angular size, median size at fixed magnitude,
derived mean surface brightness, the fitted Euclid completeness surface, and
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
