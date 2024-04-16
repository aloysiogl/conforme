# Abstract

Split conformal prediction is a statistical method known for its finite-sample coverage guarantees, simplicity, and low computational cost. As such, it is suitable for predicting uncertainty regions in time series forecasting. However, in the context of multi-horizon forecasting, the current literature lacks conformal methods that produce efficient intervals and have low computational cost.

Building on the foundation of split conformal prediction and one of its most prominent extensions to multi-horizon time series forecasting (CF-RNN), we introduce ConForME, a method that leverages the time dependence within time series to construct efficient multi-horizon prediction intervals with probabilistic joint coverage guarantees. We prove its validity and support our claims with experiments on both synthetic and real-world data. Across all instances, our method outperforms CF-RNN in terms of mean, min, and max interval sizes over the entire prediction horizon, achieving improvements of up to 52\%. The experiments also suggest that these improvements can be further increased by extending the prediction horizon and through hyperparameter optimization.

## Keywords

Split conformal prediction, multi-horizon time series forecasting, uncertainty quantification.
