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

The COSMOS prior is an empirical joint distribution

\[
p(m_{\rm F814W},z,\log M_\star,R_e),
\]

restricted to `generator_ready` rows with finite combined bulge+disc
circularized (R_e). A cross-validated conditional categorical model maps
stellar mass to TNG morphology. F814W-to-VIS magnitudes follow the fitted
affine relation with Gaussian scatter, and detection is a Bernoulli draw from
the fitted logistic completeness. COSMOS2025 motivates the measured catalogue
quantities and selection ([COSMOS2025](https://arxiv.org/abs/2506.03243)).
The morphology kernel, supported mass range, cuts, and numerical tolerances are
measured validation decisions, not literature facts. Extreme Deconvolution is
the reference for likelihood fitting with noisy latent distributions
([Bovy et al. 2011](https://arxiv.org/abs/0905.2979)).

## Sizes and TNG images

Each COSMOS circularized combined B+D \(R_e\) is paired directly with the
measured centered VIS curve-of-growth radius of the selected TNG frame. One
spatial resampling is applied to the complete registered TNG cube:

\[
s_0=\frac{R_{e,\rm COSMOS}}
          {R_{e,\rm TNG,px}\,p},
\]

where \(p\) is the output pixel scale. The same geometric transform is used for
VIS/Y/J/H, preserving the native TNG inter-band ratios. After cropping, one
VIS-anchored scalar multiplies the entire cube; the COSMOS-conditioned path
has no per-band colour correction. The 1600-pixel atlas frame side is not part
of this calculation. The final cropped VIS stamp is remeasured and the scale
is refined until the residual is within 5% or 0.5 output pixel, whichever is
larger. The effective-radius convention is the standard Sérsic/curve of growth
definition ([Graham & Driver 2005](https://arxiv.org/abs/astro-ph/0503176)).
The TNG50-SKIRT atlas and its effective-radius analysis provide the simulated
multiband frames and comparison context ([TNG50-SKIRT Atlas](https://arxiv.org/abs/2401.04224),
[effective-radius analysis](https://arxiv.org/abs/2401.04225)).

Atlas images are scanned only on FASRC. The resulting
`tng_atlas_parameters.csv` has one row per subhalo and orientation, joining
the measured native VIS half-light radius to stellar mass, SFR, halo mass,
and the TNG catalogue stellar half-mass radius. A fingerprinted JSON sidecar
identifies the exact remote atlas inventory and measurement algorithm. Local
population fitting reads this compact table only; it neither downloads nor
plots the SKIRT frames.

## What is fitted versus imposed

The empirical COSMOS and Gaia distributions, brightness transfer, completeness,
TNG morphology kernel, and stellar flux-error model are fitted or measured from
their respective artifacts. Homogeneous positions, Poisson counts, the
five-orientation TNG inventory, and the 5%/0.5-pixel radius tolerance are
simulation or validation choices. Euclid Q1 MER photometry motivates the
four-band measurement context ([Euclid Q1 MER catalogue](https://arxiv.org/abs/2503.15305)).
