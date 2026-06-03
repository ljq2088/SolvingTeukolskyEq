# Spectral surrogate lessons

- Use physical coordinate `y=2r_+/r-1`; horizon is `y=1`, infinity is `y=-1`.
- Local spectral solves are reliable; direct residual optimization from ordinary initialization is not.
- For neural surrogates, output low-order Chebyshev coefficients only and set high modes to zero.
- Initialize the final output bias from the center-parameter teacher coefficients, especially for large-amplitude bases such as `u_up`.
- Check both function error and relative ODE residual; small function error can still produce large residual after spectral differentiation.
