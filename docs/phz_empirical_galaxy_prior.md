# Empirical PHZ galaxy draw prior

The default field-galaxy draw is deliberately an observed-space baseline. It
does not fit a latent luminosity function or a size-evolution relation.

For every usable MER+PHZ catalogue row, define

\[
w_i = 1-P_{{\rm point},i}, \qquad
r_i = 0.1\,a_i\sqrt{1-e_i}, \qquad
m_{{\rm Kron},i}=23.90-2.5\log_{10}
  \left(F_{{\rm detection,total},i}/{\rm \mu Jy}\right).
\]

Here (a_i) is `SEMIMAJOR_AXIS` in VIS pixels and (e_i) is
`ELLIPTICITY`. The radius is therefore a circularized MER detection proxy, not
a fitted half-light radius. Each selected object contributes

\[
p(z,m,r) \;\mathrel{+}=\;
\frac{w_i}{A}\,p_i(z)\,
\mathbf{1}_{m_i\in m}\,\mathbf{1}_{\log r_i\in\log r},
\]

where (p_i(z)) is its normalized PHZ redshift PDF and (A) is the total
queried area. There is no magnitude or radius clipping, so the observed bright
and large tails remain in the sampling grid. Positions are drawn from a
homogeneous Poisson process at the grid's integrated surface density.

For a sampled cell, the generator draws (z), (m_{\rm Kron}), and (r)
uniformly inside that cell, selects a diversity-balanced TNG morphology, first
resizes it to (r), and only then applies one scalar to all four bands so the
integrated clean VIS flux equals the sampled Kron flux. This last equality is a
brightness-anchor convention. It is not yet a claim that a post-PSF, noisy
source-extraction run would remeasure exactly the same Kron magnitude.

The population artifact is marked `validated: false` until rendered fields are
run through a Kron measurement and compared with the MER distribution.
