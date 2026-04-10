# Platform Trial Meta-Analysis

Browser-based meta-analysis tool for adaptive platform trials (RECOVERY, REMAP-CAP, etc.) that share control arms, with proper variance-covariance handling via GLS pooling.

## Features

- **Shared control arm covariance**: Multi-arm platform trials introduce correlation between treatment comparisons. Off-diagonal covariance = tau^2/2 per the multi-arm trial adjustment.
- **GLS pooling**: Block-diagonal covariance matrix with Generalized Least Squares, not naive inverse-variance.
- **Non-concurrent control adjustment**: Temporal drift correction when treatment arms enter/leave at different times.
- **Platform timeline**: Gantt-style SVG visualization of enrollment periods across platforms.
- **Forest plot**: Treatment vs control comparisons grouped by platform, displayed as HR with 95% CI.
- **Covariance matrix display**: Visualize the within-trial V matrix for multi-arm platforms.
- **Export**: CSV results and SVG visualizations.

## Input Modes

**Per-Arm Data** (recommended):
```
Platform,Arm,ArmType,Effect_vs_Control,SE,StartMonth,EndMonth,N
```

**Pre-Computed**:
```
Platform,Treatment,LogEffect,SE,ControlN,TreatmentN,SharedControlArms
```

## Statistical Methods

- Effects pooled on log scale (log HR), displayed as HR
- Between-trial heterogeneity: DerSimonian-Laird (auto) or manual tau^2
- Within-trial covariance: SE^2 + tau^2 on diagonal, tau^2/2 off-diagonal for shared control
- Matrix inversion: closed-form for 1x1, 2x2, 3x3; Gauss-Jordan for larger
- Positive-definiteness enforced via Higham-style nearPD

## Demo Data

4 platform trials: RECOVERY (3 arms), REMAP-CAP (2 arms), SOLIDARITY (1 arm), PANORAMIC (1 arm).

## Testing

```bash
cd C:\Models\PlatformTrialMA
python -m pytest test_app.py -v
```

21 tests covering covariance structure, pooling, NCA, visualization, edge cases, and export.

## Usage

Open `index.html` in any modern browser. No dependencies, no CDN, fully offline.

## Author

Mahmood Ahmad, Tahir Heart Institute
