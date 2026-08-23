## Does the paid-search retention gap survive platform?

| platform | paid_search D7+ | organic D7+ | gap | 95% CI | significant |
|---|---|---|---|---|---|
| **all** | | | -12.1 pp | [-13.0, -11.2] | yes |
| android | 0.695 | 0.824 | -12.9 pp | [-14.4, -11.4] | yes |
| ios | 0.712 | 0.825 | -11.3 pp | [-12.7, -9.9] | yes |
| web | 0.704 | 0.826 | -12.2 pp | [-13.9, -10.5] | yes |

the gap is present and significant inside every platform, so it is not a platform-mix artifact

## Simpson check

no Simpson reversal: every aggregate channel comparison keeps its sign inside every platform
(10 channel pairs checked, 0 reversals)

## Cohort over cohort

18 weeks reported, 0 excluded as thin.
Gap drift over the window: -2.93 points (slope -0.1723 pp/week).

no significant cohort trend (slope -0.17 pp/week, z=-1.9); the gap is a standing quality difference, not a live regression

## Segmentation on unvalidated data

### metric: `d1_exact`

on d1_exact, the uncleaned warehouse reports a -4.7 point ios effect (significant) where the cleaned one reports -0.0. The -4.7 point difference is the timezone bug, not the platform. Segmentation on unvalidated data produces confident findings, not noisy ones.

| warehouse | android | ios | web |
|---|---|---|---|
| clean | 0.4182 | 0.4167 | 0.4158 |
| uncleaned | 0.4182 | 0.3697 | 0.4158 |

### metric: `d7_plus`

on d7_plus, the uncleaned warehouse reports a +0.3 point ios effect (not significant) where the cleaned one reports +0.4. The -0.1 point difference is the timezone bug, not the platform. Segmentation on unvalidated data produces confident findings, not noisy ones.

| warehouse | android | ios | web |
|---|---|---|---|
| clean | 0.7736 | 0.7798 | 0.7788 |
| uncleaned | 0.7736 | 0.7793 | 0.7788 |

