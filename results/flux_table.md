## Run A - noiseless, cleaned deep (threshold_sigma = 0.05)

Does the algorithm conserve flux when nothing else is in the way?

| Source | l (")| m (")| Input (Jy) | Recovered (Jy) | Error (%) |
|---|---|---|---|---|---|
| S1 | +0.0 | +0.0 | 1.000 | 1.000 | -0.04 |
| S2 | +90.0 | +60.0 | 0.500 | 0.500 | -0.08 |
| S3 | -120.0 | +45.0 | 0.250 | 0.250 | -0.16 |
| S4 | +60.0 | -109.5 | 0.125 | 0.125 | -0.31 |
| S5 | -75.0 | -79.5 | 0.060 | 0.060 | -0.64 |
| **Total** | | | **1.935** | **1.933** | **-0.10** |

## Run B - with thermal noise, cleaned to 3 sigma

What you would actually do with real data, and what it costs.

| Source | l (")| m (")| Input (Jy) | Recovered (Jy) | Error (%) |
|---|---|---|---|---|---|
| S1 | +0.0 | +0.0 | 1.000 | 0.977 | -2.26 |
| S2 | +90.0 | +60.0 | 0.500 | 0.475 | -4.93 |
| S3 | -120.0 | +45.0 | 0.250 | 0.224 | -10.35 |
| S4 | +60.0 | -109.5 | 0.125 | 0.102 | -18.12 |
| S5 | -75.0 | -79.5 | 0.060 | 0.036 | -40.48 |
| **Total** | | | **1.935** | **1.815** | **-6.20** |

## Run C - extended emission

Input 1.400 Jy, CLEAN model 0.944 Jy (67.4 % recovered).