# Gradient checkpointing
Gradient checkpointing for graph mode execution in Tensorflow 2

This is a standalone version extracted from the original implementation in [tf-slim](https://github.com/google-research/tf-slim/blob/a62dc893de5e46e6f2e9ec24a74b2abce026307a/tf_slim/layers/rev_block_lib.py).

If using eager execution, use [tf.recompute_grad](https://www.tensorflow.org/api_docs/python/tf/recompute_grad).

For more information on recomputing gradients between graph nodes during backpropagation, see [the original gradient checkpointing repository](https://github.com/cybertronai/gradient-checkpointing).

***

Tested with `tf-nightly==2.2.0.dev20200303` in graph mode on TPU.

Example usage for a model built with a Keras layer `call` method:

```
def call(self, x, past):
    @gradient_checkpointing.recompute_grad
    def inner(x):
        # ops go here
        return y
    return inner(x)
```

Note: Gradient checkpointing can significantly slow down training.  
