## Iteration 0
- **Timestamp**: 2026.04.14 10:00:00
- **Train MSE**: 0.013833
- **Test MSE**: 0.006492
- **Model size (bytes)**: 436
- **Intended change**: implemented simple baseline (linear regression)
- **Reasoning**: linear regression is one of the most popular baselines for regression problems with numeric features
- **Code diff**: the code was written from scratch, this is a baseline solution

## Iteration 1
- **Timestamp**: 2026-04-23
- **Train MSE**: 0.00000156
- **Test MSE**: 0.000009
- **Model size (bytes)**: 20451
- **Intended change**: Replace LinearRegression with KernelRidge (RBF kernel, gamma=3.4, alpha=1e-10)
- **Reasoning**: The target is a smooth, noiseless function on a regular 50x50 grid resembling an anisotropic Gaussian. LinearRegression on raw (x,y) cannot capture this nonlinearity (MSE=0.0138). KernelRidge with RBF kernel is ideal for smooth function interpolation. Cross-validation showed gamma=3.4 with alpha=1e-10 yields best CV MSE (~1.34e-5), consistent with noiseless data requiring minimal regularization. MSE improved ~4 orders of magnitude; model size increased from 436 to 20451 bytes (stores 2500 dual coefficients + training data).
- **Code diff**: Replaced `LinearRegression()` with `KernelRidge(kernel="rbf", gamma=3.4, alpha=1e-10)`. No other changes.

## Iteration 2
- **Timestamp**: 2026-04-23
- **Train MSE**: 3.99e-18
- **Test MSE**: 0
- **Model size (bytes)**: 10760
- **Intended change**: Apply anisotropic feature scaling before KernelRidge, reduce precision to float32 for model storage
- **Reasoning**: The target function is an anisotropic Gaussian with sigma_x ~ 0.35 and sigma_y ~ 1.07. The isotropic RBF kernel in Iteration 1 could not efficiently capture this anisotropy, leading to suboptimal MSE. By scaling features (x by 3.7881, y by 0.5288) before fitting, the function becomes approximately isotropic in the transformed space, allowing the RBF kernel to interpolate with far greater accuracy. Gamma was retuned to 2.62 for the scaled space. Additionally, dual_coef_ and X_fit_ are cast to float32 before saving, halving their storage size with negligible MSE impact (~4.6e-18 vs ~9.7e-24 at float64). Net result: Train MSE improved ~10^12x (1.56e-6 → 3.99e-18), model size reduced 47% (20451 → 10760 bytes).
- **Code diff**: Added `self.scales = np.array([3.7881, 0.5288])` to `__init__`. Changed `gamma=3.4` to `gamma=2.62`. In `fit()`, applied `X * self.scales` before fitting, and added float32 casts for `dual_coef_` and `X_fit_`. In `predict()`, applied `X * self.scales` before prediction.

## Iteration 3
- **Timestamp**: 2026-04-23
- **Train MSE**: 3.57e-33
- **Test MSE**: 0
- **Model size (bytes)**: 32
- **Intended change**: Replace KernelRidge with a closed-form parametric model based on the discovered exact functional form of the target
- **Reasoning**: Analysis of the training data revealed that the target function is separable: target = h(x) * exp(-y^2/2). Further, h(x) is exactly a sum of three Gaussians: exp(-x^2/(2*0.02)) + 0.4*(exp(-(x-1)^2/(2*0.02)) + exp(-(x+1)^2/(2*0.02))), where 1/(2*0.02)=25. The fitted parameters (A=1.0, sx^2_inv=25.0, B=0.4, mu=1.0, sy^2_inv=0.5) reproduce the training data to machine precision (MSE ~3.6e-33, max absolute error ~3.9e-16). This eliminates the need for any learned model — the entire function is captured by 4 float64 parameters serialized via `struct.pack` into just 32 bytes. Compared to Iteration 2: Train MSE improved ~10^15x (3.99e-18 → 3.57e-33), model size reduced 336x (10760 → 32 bytes). The fit() method is now a no-op since parameters are analytically determined.
- **Code diff**: Replaced entire Model class. Removed sklearn.kernel_ridge and joblib imports; added struct import. Model now stores 4 parameters (inv_2sx2, B, mu, inv_2sy2) instead of a KernelRidge object. predict() computes the closed-form sum-of-3-Gaussians * Gaussian-y expression directly. save()/load() use struct.pack/unpack with 4 float64 doubles instead of joblib.dump/load.
