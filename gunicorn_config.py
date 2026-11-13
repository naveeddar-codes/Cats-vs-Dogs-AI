# CRITICAL for Render Free Tier (512MB RAM):
# 1 single worker prevents multiple copies of TensorFlow model in memory.
workers = 1
threads = 4
worker_class = "gthread"
timeout = 120