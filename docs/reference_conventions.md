# Reference conventions for `s=-2, l=m=2`

This note freezes the benchmark roles used by the high-accuracy Teukolsky
solver branch.

## Radial solution

The target solution is the horizon-normalized ingoing homogeneous radial mode
`R_in(r; a, omega)` on `r in [r_+, infinity)`.

The project compactification is

```text
x = r_+ / r
y = 2 x - 1
```

The current horizon-side reduced shape is

```text
R_in = P_H(r,a,omega) * h2(a,omega) * S(x;a,omega)
S(x) = g(x) * (h1(x) * f(y) + 1) + 1
```

with hard constraints `S(1)=1` and `S_x(1)=c_H=-A0/A1`.

## Infinity amplitudes

At infinity the convention used by `utils/amplitude.py` is

```text
R_in(r) ~ B_inc * A_down(r) + B_ref * A_up(r)
A_down(r) = r^{-1} * exp(-i omega r_*)
A_up(r)   = r^3    * exp(+i omega r_*)
```

`B_trans = 1` in the horizon-normalized convention.

## Benchmark roles

- `pybhpt`: trusted primarily for `R_in(r)` at small/moderate `omega`; it is not
  treated as a trusted source for `B_inc/B_ref` in this project.
- `MMA`: trusted for small/moderate `omega` amplitudes and convention checks.
- `GSN`: trusted for larger `omega` amplitudes and high-frequency checks.
- `omega ~= 0.1`: mandatory overlap band for pybhpt/MMA/GSN cross-validation.

## Acceptance metric

Final acceptance is the relative equation residual

```text
|residual| / max(|second-order term|, |first-order term|, |zeroth-order term|)
```

where the denominator terms include the solution itself and are not divided by
external normalization constants.

Benchmark/anchor losses are training aids only. They are not the final truth
criterion.
