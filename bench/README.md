# bench/

The paid cloud benchmark launchers were removed from this Spark-first checkout.

Use `train_local.py` and `Dockerfile.spark` for local DGX Spark training. Add new
benchmark scripts here only if they run locally on the Spark without paid cloud
GPU launchers.
