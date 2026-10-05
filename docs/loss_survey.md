# Loss survey for observed → ideal deconvolution

The target is a sparse ideal image: most pixels are exactly zero, and the sources run
from far below the noise (≈95% of them have S/N < 3, see the Cramér–Rao map) to bright
cores ~10⁵× the faint ones. We train in asinh-normalized space. The 2×2 ablation showed
that the loss sets total faint flux (MSE recovers far more than L1) and depth sets
per-star accuracy. So the question for a loss is mostly **which statistic of the
per-pixel flux distribution it estimates**, and whether that statistic survives the
asinh inverse:

- L1 estimates the median. For a pixel that holds a faint source in only some of the
  plausible worlds, the median is zero, so faint flux is lost.
- MSE estimates the mean, but of asinh(flux). asinh is concave on positive flux, so by
  Jensen the inverse of that mean is below the mean flux.
- An unbiased total flux needs the mean *in flux space*.

This page condenses three literature surveys (restoration losses, point-source
deconvolution losses, likelihood and flux-space losses). Citations were checked against
arXiv or the publisher unless marked [unverified]. "Selected" means part of the ESPCN
screen in `config/loss_screen/`; the implementations are in `core/losses.py`.

## Screened

| Name | Source | Statistic estimated | Needs | Fit |
|---|---|---|---|---|
| `l1` | Zhao et al., IEEE TCI 2017, [arXiv:1511.08861](https://arxiv.org/abs/1511.08861) | median | nothing | reference; loses faint flux |
| `mse` | same | mean of asinh(flux) | nothing | reference; biased low after inverse |
| `huber` | Huber, Ann. Math. Stat. 35:73 (1964), [doi:10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732) | mean for errors < δ, median above | δ | middle ground: faint stars in the L2 regime, bright-core errors in L1 |
| `flux_mse` | standard χ² (no single paper) | mean flux | asinh inverse inside the loss | unbiased by construction; bright cores dominate the gradient |
| `flux_relative_mse` | Noise2Noise, Lehtinen et al., ICML 2018, [arXiv:1803.04189](https://arxiv.org/abs/1803.04189) | mean flux | stop-gradient denominator, ε | mean fixed point with bright pixels down-weighted; built for HDR Monte-Carlo images |
| `blurred_mse` | Deep-STORM, Nehme et al., Optica 5:458 (2018), [arXiv:1801.09631](https://arxiv.org/abs/1801.09631) | mean of the blurred map | Gaussian g | same target type (point sources on a grid); a 1 px shift costs little |
| `blurred_mse_sparse` | same, with its λ‖ŷ‖₁ term | blurred mean, spurious flux suppressed | g, λ | the L1 term counters our 1% → 3.5% spurious-flux growth under MSE, but can push faint flux to zero again |
| `beta_nll` | Seitzer et al., ICLR 2022, [arXiv:2203.09168](https://arxiv.org/abs/2203.09168); Gaussian NLL from Nix & Weigend 1994, [doi:10.1109/ICNN.1994.374138](https://doi.org/10.1109/ICNN.1994.374138), Kendall & Gal, NeurIPS 2017, [arXiv:1703.04977](https://arxiv.org/abs/1703.04977) | mean and σ of asinh(flux); flux mean via E[sinh Y] = sinh(μ)e^{s²/2} | 2nd output channel (log σ²), β | β fixes plain NLL's habit of inflating σ on hard (faint) pixels; the flux correction is exact only if asinh(flux) is Gaussian |
| `tweedie` | compound Poisson–gamma deviance, 1 < p < 2; no canonical DL paper. DL uses: [arXiv:2306.09882](https://arxiv.org/abs/2306.09882), [arXiv:2505.06445](https://arxiv.org/abs/2505.06445), [arXiv:2406.16206](https://arxiv.org/abs/2406.16206) | mean flux | log-link output, p | point mass at zero plus a skewed positive part, as our target has |

Fixed hyperparameters and why are documented next to each loss in `core/losses.py`.
They are not swept: 9 losses × 150k iterations is the budget, and only the winners
get a sweep. L1 and MSE are also run at seed 43 to measure run-to-run noise.

## Not screened

**Drop-in losses that estimate the same thing as a screened one**

| Loss | Source | Statistic | Why not |
|---|---|---|---|
| Charbonnier √(x²+ε²) | LapSRN, Lai et al., CVPR 2017, [arXiv:1704.03915](https://arxiv.org/abs/1704.03915) | median (ε ~ 1e-3) | recreates L1; with a large ε it is Huber |
| Smooth L1 | Fast R-CNN, Girshick, ICCV 2015, [arXiv:1504.08083](https://arxiv.org/abs/1504.08083) | Huber with δ = 1 | covered by `huber` |
| log-cosh | Saleh & Saleh 2022, [arXiv:2208.04564](https://arxiv.org/abs/2208.04564) | between mean and median | same role as Huber, no advantage |
| Barron's adaptive robust loss | Barron, CVPR 2019, [arXiv:1701.03077](https://arxiv.org/abs/1701.03077) | mean at α=2, ~median at α=1 | learning α drifts toward robust (median-like); a fixed α in 1.3–2 is Huber-like. Candidate for a later sweep |
| Pinball / quantile | Koenker & Bassett, Econometrica 46:33 (1978), [doi:10.2307/1913643](https://doi.org/10.2307/1913643) | τ-quantile | τ > 0.5 offsets the zero bias but inflates the background; median-like at 0.5 |
| L1 + λ·L2 | common practice | between median and mean | same as Huber with λ as the knob |
| Laplace NLL | Kendall & Gal 2017 (depth) | median | same faint-flux loss as L1 |
| Poisson NLL | standard GLM | mean flux | assumes variance = mean, which flux does not obey; Tweedie generalises it. Related: [arXiv:2406.09262](https://arxiv.org/abs/2406.09262) [venue unverified] |

**Losses that don't constrain flux**

| Loss | Source | Why not |
|---|---|---|
| SSIM / MS-SSIM, "Mix" | Zhao et al. 2017 (above) | normalises away brightness and contrast; variance terms unstable on mostly-zero targets |
| FFT-L1 (MSFR) | MIMO-UNet, Cho et al., ICCV 2021, [arXiv:2108.05054](https://arxiv.org/abs/2108.05054) | still penalises a 1 px shift (as phase); FFT-L2 equals pixel L2 by Parseval |
| Focal Frequency Loss | Jiang et al., ICCV 2021, [arXiv:2012.12821](https://arxiv.org/abs/2012.12821) | global, for generative models; nothing for per-star photometry |
| Frequency Distribution Loss | Ni et al., CVPR 2024, [arXiv:2402.18192](https://arxiv.org/abs/2402.18192) | compares deep-feature spectra, too indirect for flux |
| Wavelet / guided-frequency losses | [arXiv:2404.11273](https://arxiv.org/abs/2404.11273), [arXiv:2402.19215](https://arxiv.org/abs/2402.19215), [arXiv:2309.15563](https://arxiv.org/abs/2309.15563) [details unverified] | as above |
| VGG perceptual | Johnson et al., ECCV 2016, [arXiv:1603.08155](https://arxiv.org/abs/1603.08155) | features not proportional to flux; rewards invented structure. Blau & Michaeli, CVPR 2018, [arXiv:1711.06077](https://arxiv.org/abs/1711.06077): perception–distortion trade-off |

**Target-dependent weighting (biases the estimate)**

| Loss | Source | Why not |
|---|---|---|
| Focal / Focal-R | Lin et al., ICCV 2017, [arXiv:1708.02002](https://arxiv.org/abs/1708.02002); Yang et al., ICML 2021, [arXiv:2102.09554](https://arxiv.org/abs/2102.09554) | skews the regression estimate on purpose; only useful on a presence head |
| Balanced MSE for precipitation | Shi et al., NeurIPS 2017, [arXiv:1706.03458](https://arxiv.org/abs/1706.03458) | weights depend on the target, so the estimate is biased toward heavy values |
| χ² with the σ_CR map as weight | standard | σ_CR includes the photon noise of the local signal, so it depends on the target, and a target-dependent weight biases the estimate; it is also ~uniform (1st–99th pct 0.0124–0.0275), so `flux_mse` uses one uniform weight |

**Post-hoc corrections of the asinh bias**

| Method | Source | Why not |
|---|---|---|
| Smearing estimator | Duan, JASA 78:605 (1983), [doi:10.1080/01621459.1983.10478017](https://doi.org/10.1080/01621459.1983.10478017) | assumes the same residual distribution everywhere; faint and bright pixels differ. Fine as a no-retraining baseline later |
| TranSUN | Yu et al., NeurIPS 2025, [arXiv:2505.13881](https://arxiv.org/abs/2505.13881) | learns a bias-correction branch jointly; only the abstract read. Worth reading if a flux-space loss underperforms |
| μ-law HDR loss | Kalantari & Ramamoorthi, ACM TOG 36(4) 2017, [doi:10.1145/3072959.3073609](https://doi.org/10.1145/3072959.3073609) | L2 after a tone curve: the same transformed-space bias we already have |

**Distributional and set-prediction heads (a separate project after the screen)**

These model "is a source here" separately from "how bright", which is the principled
fix for faint stars, but each needs a multi-channel head, its own decoder back to an
image, and more tuning than a loss swap.

| Method | Source | Statistic / head | Notes |
|---|---|---|---|
| Hurdle (presence + lognormal/gamma flux) | Vandal et al., KDD 2018, [arXiv:1802.04742](https://arxiv.org/abs/1802.04742) | full distribution; 2–3 channels | beat a Gaussian on rainfall; flux head sees few samples |
| Mixture density network | Bishop 1994, [NCRG/94/004](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) | full distribution; 3K channels | K = 2 is a soft hurdle; component collapse |
| Heteroscedastic fixes | Stirn et al., AISTATS 2023, [arXiv:2212.09184](https://arxiv.org/abs/2212.09184); Immer et al., NeurIPS 2023, [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a901d5540789a086ee0881a82211b63d-Abstract-Conference.html) | mean + σ | alternatives to β-NLL if it misbehaves |
| DECODE | Speiser et al., Nat. Methods 18:1082 (2021), [doi:10.1038/s41592-021-01236-x](https://doi.org/10.1038/s41592-021-01236-x) | count + Gaussian-mixture NLL over (p, Δx, Δy, flux, σs); ~7 channels in 2D | flux regressed separately from detection; GPL-3.0 ([code](https://github.com/TuragaLab/DECODE)). MIT reimplementation in LiteLoc, Nat. Commun. 2025, [doi:10.1038/s41467-025-62662-5](https://doi.org/10.1038/s41467-025-62662-5). FD-DeepLoc, Nat. Methods 2023, [doi:10.1038/s41592-023-01775-5](https://doi.org/10.1038/s41592-023-01775-5), adds a field-varying PSF |
| DeepSTORM3D | Nehme et al., Nat. Methods 17:734 (2020), [arXiv:1906.09957](https://arxiv.org/abs/1906.09957) | blurred MSE + Dice | Dice helps detection in dense fields; drop-in, could follow `blurred_mse` |
| SHOT | Seailles et al., ICLR 2026 [venue unverified], [arXiv:2512.10683](https://arxiv.org/abs/2512.10683) | Sinkhorn OT matching on (x, y, flux) + detection BCE | no double penalty for a shifted star; AGPL-3.0 |
| StarNet / BLISS | Liu, McAuliffe, Regier, JMLR 24 (2023), [arXiv:2102.02409](https://arxiv.org/abs/2102.02409) | amortized VI (forward KL), catalog head | built for crowded-field stellar photometry (M2); MIT ([BLISS](https://github.com/prob-ml/bliss)) |
| Sinkhorn divergence on the image | Feydy et al., AISTATS 2019, [arXiv:1810.08278](https://arxiv.org/abs/1810.08278) [unverified]; GeomLoss (MIT) | unbalanced OT between predicted and true flux measures | no head change, but costly on dense images |

**Astronomy deconvolution papers seen** (none targets faint-flux bias): Tikhonet,
Sureau et al., A&A 641:A67 (2020), [arXiv:1911.00443](https://arxiv.org/abs/1911.00443)
(MSE); Akhaury et al., Front. Astron. Space Sci. 2022,
[doi:10.3389/fspas.2022.1001043](https://doi.org/10.3389/fspas.2022.1001043) (MSE;
flux measured, not enforced); ShapeNet, Nammour et al., A&A 663:A69 (2022),
[arXiv:2203.07412](https://arxiv.org/abs/2203.07412) (L2 + shape moments; says L2 does
not preserve flux); source-weighted Huber/SSIM composites,
[arXiv:2512.13353](https://arxiv.org/abs/2512.13353) (heuristic); DeepSource,
[arXiv:1807.02701](https://arxiv.org/abs/1807.02701), and PNet,
[arXiv:2106.14349](https://arxiv.org/abs/2106.14349) (losses not checked);
Rawson & Hultgren, EUSIPCO 2022, [arXiv:2202.05354](https://arxiv.org/abs/2202.05354)
(optimal-transport SR, not a CNN loss).
