# E156 Protocol: Platform Trial Meta-Analysis

**Project**: PlatformTrialMA
**Start date**: 2026-04-09
**Status**: v1.0

## E156 Body (CURRENT)

Adaptive platform trials sharing control arms introduce within-trial correlation that standard pairwise meta-analysis ignores, potentially biasing pooled estimates and confidence intervals. We built a browser-based tool analyzing per-arm data from platform trials (RECOVERY, REMAP-CAP, SOLIDARITY, PANORAMIC) using generalized least squares with block-diagonal covariance matrices where off-diagonal elements equal tau-squared over two for shared-control comparisons. The engine constructs K-by-K variance-covariance blocks per platform, estimates between-trial heterogeneity via DerSimonian-Laird, and pools via GLS with matrix inversion through closed-form solutions or Gauss-Jordan elimination. With manual tau-squared of 0.01, the GLS-adjusted standard error diverged from naive inverse-variance pooling, confirming that ignoring shared-control correlation distorts precision estimates. Temporal drift correction for non-concurrent enrollment changed REMAP-CAP effect estimates where arms entered at different months. The tool correctly degenerates to standard pairwise meta-analysis for independent two-arm trials while handling multi-arm platforms with proper covariance structure. All 21 validation tests pass including covariance verification, matrix algebra, and edge cases for single-platform and two-arm scenarios.

## Dashboard

GitHub Pages: `https://mahmood726-cyber.github.io/PlatformTrialMA/`

## Validation

- 21 Selenium tests (pytest)
- Covariance: off-diagonal = tau^2/2 verified for 3-arm RECOVERY
- Matrix inverse identity verified for 2x2 and 3x3
- GLS vs naive IV divergence confirmed with tau^2 > 0
