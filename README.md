# Gradient checkpointing
Gradient checkpointing for graph mode execution in Tensorflow 2

If using eager execution, the [tf.recompute_grad](https://www.tensorflow.org/api_docs/python/tf/recompute_grad) functionality should be used.

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
