"""F1-F3: wide or sharp, never both (until Euclid). Survey area against PSF FWHM, log-log, deck style.
Values (round, to be checked before the talk): SDSS 14,555 deg2 / 1.4"; DES 5,000 / 0.9"; HSC-SSP Wide 1,400 /
0.6"; LSST 18,000 / 0.7"; Hubble COSMOS 1.7 / 0.09"; JWST COSMOS-Web 0.54 / 0.05"; Euclid Wide 14,000 /
0.15" (VIS, measured on Q1)."""
import matplotlib.pyplot as plt

from style import BG, CORAL, MONO, MUTED, SANS, TEAL, TXT, save

pts = [("SDSS", 1.4, 14555, "g"), ("DES", 0.9, 5000, "g"), ("HSC Wide", 0.6, 1400, "g"),
       ("Rubin LSST", 0.7, 18000, "g"), ("Hubble COSMOS", 0.09, 1.7, "s"),
       ("JWST COSMOS-Web", 0.05, 0.54, "s"), ("Euclid Wide", 0.15, 14000, "e")]
fig = plt.figure(figsize=(19.2, 10.8), dpi=200, facecolor=BG)
ax = fig.add_axes([0.11, 0.13, 0.82, 0.78], facecolor=BG)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(2.0, 0.035); ax.set_ylim(0.2, 60000)
col = {"g": MUTED, "s": "#7fb2ff", "e": CORAL}
for name, fwhm, area, k in pts:
    ax.scatter(fwhm, area, s=900 if k == "e" else 420, color=col[k], zorder=3, edgecolor="none")
    left = name.startswith("JWST")
    dx = (1 / 1.14) if left else (1.12 if k != "e" else 1.18)
    ax.text(fwhm / dx, area, name, color=TXT if k != "g" else MUTED, fontsize=26, family=SANS,
            weight="bold" if k == "e" else "medium", va="center", ha="right" if left else "left")
ax.text(1.85, 35000, "ground: wide, blurred", color=MUTED, fontsize=20, family=MONO, va="center")
ax.text(0.2, 4.5, "space: sharp, tiny", color="#7fb2ff", fontsize=20, family=MONO, va="center")
for s in ax.spines.values():
    s.set_color("#2c3340")
ax.tick_params(colors=MUTED, labelsize=20, which="both")
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_family(SANS)
ax.set_xticks([1, 0.5, 0.2, 0.1, 0.05]); ax.set_xticklabels(["1″", "0.5″", "0.2″", "0.1″", "0.05″"])
ax.set_yticks([1, 10, 100, 1000, 10000]); ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
ax.set_xlabel("sharper →   (PSF FWHM)", color=TXT, fontsize=24, family=SANS, labelpad=14)
ax.set_ylabel("area surveyed (deg²)", color=TXT, fontsize=24, family=SANS, labelpad=14)
ax.grid(color="#1f2531", lw=1, which="major")
save(fig, "F1_wide_or_sharp.png")
print("ok")
