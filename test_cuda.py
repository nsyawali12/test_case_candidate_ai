import tensorflow as tf
# gpu_available = tf.config.list_physical_devices('GPU')

is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)