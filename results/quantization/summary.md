# Quantization with CircuitKIT (mukeshkbj/smallprint-tiny-aya-water-3b)

Circuit found with EAP-IG on 64 fair/unfair label-swap pairs; `circuit` keeps the top 15% of layers in bf16, `random` keeps the same number chosen at random, `uniform` keeps none. False alarms and missed unfair clauses are lower-is-better; F1 alone hides a shift between them.

### lexglue

| Version | Weights (GB) | unfair F1 | macro-F1 | false alarms | missed unfair | clauses/s |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 6.70 | 0.788 | 0.760 | 0.043 | 0.116 | 5.3 |
| int4_uniform | 2.48 | 0.805 | 0.757 | 0.015 | 0.244 | 2.1 |
| int4_circuit_0.15 | 3.07 | 0.810 | 0.753 | 0.031 | 0.145 | 2.3 |
| int4_random_0.15 | 3.07 | 0.781 | 0.730 | 0.043 | 0.128 | 2.3 |

### claudette_ml

| Version | Weights (GB) | unfair F1 | macro-F1 | false alarms | missed unfair | clauses/s |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 6.70 | 0.619 | 0.506 | 0.079 | 0.135 | 5.3 |
| int4_uniform | 2.48 | 0.693 | 0.537 | 0.029 | 0.294 | 2.1 |
| int4_circuit_0.15 | 3.07 | 0.624 | 0.501 | 0.067 | 0.191 | 2.3 |
| int4_random_0.15 | 3.07 | 0.602 | 0.493 | 0.072 | 0.206 | 2.3 |

