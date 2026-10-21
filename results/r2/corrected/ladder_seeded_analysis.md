# Seeded execution ladder (P3 / E6) -- corrected costs, full analysis

Source: `results/r2/corrected/ladder_seeded.csv` (8 configs x seeds 1-5, all complete). Fold 2 test window. Net return = cumulative net-of-cost return over the test window. Config f is reused from P1 canonical (identical config, seed, code path). Pairing is by seed index; configs differ in env/exploration settings, so no init-tensor equality across configs is claimed. n = 5 per config: CIs are wide and p-values are descriptive, uncorrected for multiplicity.

## 1. Per-config net return

| cfg | description | mean | std | 95% t-CI | s1 | s2 | s3 | s4 | s5 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| a | daily, no band, no smoothing, no cap | -0.053 | 0.499 | [-0.673, +0.567] | -0.772 | +0.568 | +0.036 | +0.160 | -0.257 |
| b | a + smoothing 0.1 | +0.007 | 0.194 | [-0.234, +0.248] | +0.052 | +0.193 | -0.128 | -0.255 | +0.172 |
| c | a + 2.0x cap | +0.164 | 0.149 | [-0.020, +0.348] | +0.046 | +0.217 | +0.030 | +0.133 | +0.393 |
| d | a + smoothing + cap | +0.235 | 0.299 | [-0.136, +0.607] | +0.001 | +0.572 | +0.530 | -0.072 | +0.145 |
| e | d + exploration annealing (entropy 0.05->0.005, log_std ceiling -> -1.897) | +0.181 | 0.227 | [-0.100, +0.463] | +0.407 | +0.288 | +0.258 | -0.187 | +0.141 |
| f | e + band 10% + 5-day cadence (== canonical) | +0.243 | 0.114 | [+0.102, +0.384] | +0.274 | +0.362 | +0.271 | +0.251 | +0.054 |
| g | e + band 10% only (daily decisions) | +0.248 | 0.304 | [-0.129, +0.625] | +0.569 | +0.129 | +0.538 | -0.151 | +0.154 |
| h | e + 5-day cadence only (no band) | +0.267 | 0.110 | [+0.131, +0.404] | +0.289 | +0.429 | +0.122 | +0.259 | +0.239 |

Secondary metrics (mean +- std):

| cfg | gross P&L | cost/NAV | turnover |
|---|---:|---:|---:|
| a | +0.586 +- 0.906 | +0.790 +- 0.161 | +0.363 +- 0.078 |
| b | +0.564 +- 0.345 | +0.478 +- 0.014 | +0.225 +- 0.008 |
| c | +0.870 +- 0.215 | +0.401 +- 0.080 | +0.178 +- 0.038 |
| d | +0.626 +- 0.406 | +0.236 +- 0.008 | +0.110 +- 0.004 |
| e | +0.574 +- 0.318 | +0.248 +- 0.010 | +0.115 +- 0.006 |
| f | +0.402 +- 0.145 | +0.031 +- 0.004 | +0.014 +- 0.002 |
| g | +0.510 +- 0.382 | +0.144 +- 0.012 | +0.067 +- 0.007 |
| h | +0.468 +- 0.147 | +0.059 +- 0.004 | +0.027 +- 0.002 |

## 2. Per-step paired deltas (net return, same seed)

| step | mean | SD | 95% t-CI | paired t p | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| b-a | +0.060 | 0.544 | [-0.616, +0.735] | 0.818 | +0.824 | -0.375 | -0.164 | -0.415 | +0.428 |
| c-a | +0.217 | 0.495 | [-0.398, +0.832] | 0.383 | +0.818 | -0.351 | -0.006 | -0.026 | +0.650 |
| d-a | +0.288 | 0.400 | [-0.209, +0.785] | 0.183 | +0.773 | +0.004 | +0.494 | -0.232 | +0.402 |
| e-d | -0.054 | 0.282 | [-0.404, +0.297] | 0.692 | +0.406 | -0.283 | -0.272 | -0.115 | -0.005 |

Same deltas decomposed (mean, paired): gross P&L / cost/NAV

| step | d gross P&L | d cost/NAV | d turnover |
|---|---:|---:|---:|
| b-a | -0.022 +- 0.966 | -0.312 +- 0.171 | -0.138 +- 0.083 |
| c-a | +0.284 +- 0.865 | -0.389 +- 0.164 | -0.185 +- 0.078 |
| d-a | +0.039 +- 0.733 | -0.554 +- 0.155 | -0.253 +- 0.074 |
| e-d | -0.052 +- 0.422 | +0.012 +- 0.011 | +0.005 +- 0.006 |

## 3. 2x2 factorial: band x cadence (e, f, g, h)

Cell means (net return, mean +- std, n=5):

| | daily | 5-day cadence | cadence effect within row |
|---|---:|---:|---:|
| no band | e: +0.181 +- 0.227 | h: +0.267 +- 0.110 | +0.086 |
| band 10% | g: +0.248 +- 0.304 | f: +0.243 +- 0.114 | -0.005 |
| band effect within column | +0.066 | -0.025 | |

Per-seed contrasts (paired by seed; df = 4):

| effect | definition | mean | SD | 95% t-CI | paired t p | bootstrap 95% CI | s1 | s2 | s3 | s4 | s5 |
|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| band main | ((f+g)-(e+h))/2 | +0.021 | 0.132 | [-0.143, +0.184] | 0.742 | [-0.076, +0.126] | +0.073 | -0.113 | +0.214 | +0.014 | -0.085 |
| cadence main | ((f+h)-(e+g))/2 | +0.040 | 0.269 | [-0.294, +0.374] | 0.755 | [-0.163, +0.251] | -0.207 | +0.187 | -0.201 | +0.424 | -0.001 |
| interaction | (f-h)-(g-e) | -0.091 | 0.119 | [-0.239, +0.056] | 0.161 | [-0.176, +0.011] | -0.177 | +0.093 | -0.130 | -0.044 | -0.198 |
| band @ daily | g-e | +0.066 | 0.165 | [-0.139, +0.272] | 0.419 | [-0.060, +0.192] | +0.162 | -0.159 | +0.280 | +0.036 | +0.014 |
| band @ 5d | f-h | -0.025 | 0.120 | [-0.174, +0.125] | 0.668 | [-0.117, +0.075] | -0.015 | -0.067 | +0.149 | -0.008 | -0.184 |
| cadence @ no band | h-e | +0.086 | 0.236 | [-0.207, +0.379] | 0.462 | [-0.082, +0.272] | -0.118 | +0.140 | -0.136 | +0.445 | +0.098 |
| cadence @ band | f-g | -0.005 | 0.310 | [-0.390, +0.379] | 0.971 | [-0.245, +0.234] | -0.295 | +0.233 | -0.266 | +0.402 | -0.100 |
| both vs neither | f-e | +0.061 | 0.226 | [-0.219, +0.341] | 0.578 | [-0.085, +0.260] | -0.133 | +0.074 | +0.013 | +0.438 | -0.086 |

Note: with n=5 the percentile bootstrap under-covers; the t-CI is the primary interval. Interaction = (f-h)-(g-e) = f-g-h+e, the difference of the two simple band effects (unhalved); positive means band and cadence are super-additive.

