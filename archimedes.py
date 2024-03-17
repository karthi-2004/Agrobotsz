import tensorflow.compat.v2 as tf

from keras.src.optimizers import optimizer
from keras.src.saving.object_registration import register_keras_serializable

# isort: off
from tensorflow.python.util.tf_export import keras_export


@register_keras_serializable()
@keras_export(
    "keras.optimizers.experimental.RMSprop",
    "keras.optimizers.RMSprop",
    "keras.dtensor.experimental.optimizers.RMSprop",
    v1=[],
)
class Archimedes(optimizer.Optimizer):
    def __init__(
            self,
            learning_rate=0.001,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-7,
            centered=False,
            weight_decay=None,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=100,
            jit_compile=True,
            name="Archimedes",
            **kwargs
    ):
        super().__init__(
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            name=name,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

        self._velocities = []
        for var in var_list:
            self._velocities.append(
                self.add_variable_from_reference(var, "velocity")
            )

        self._momentums = []
        if self.momentum > 0:
            for var in var_list:
                self._momentums.append(
                    self.add_variable_from_reference(var, "momentum")
                )

        self._average_gradients = []
        if self.centered:
            for var in var_list:
                self._average_gradients.append(
                    self.add_variable_from_reference(var, "average_gradient")
                )

    def update_step(self, gradient, variable):
        lr = tf.cast(self.learning_rate, variable.dtype)

        var_key = self._var_key(variable)
        velocity = self._velocities[self._index_dict[var_key]]
        momentum = None
        if self.momentum > 0:
            momentum = self._momentums[self._index_dict[var_key]]
        average_grad = None
        if self.centered:
            average_grad = self._average_gradients[self._index_dict[var_key]]

        rho = self.rho

        if isinstance(gradient, tf.IndexedSlices):
            velocity.assign(rho * velocity)
            velocity.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - rho), gradient.indices
                )
            )
            if self.centered:
                average_grad.assign(rho * average_grad)
                average_grad.scatter_add(
                    tf.IndexedSlices(
                        gradient.values * (1 - rho), gradient.indices
                    )
                )
                denominator = velocity - tf.square(average_grad) + self.epsilon
            else:
                denominator = velocity + self.epsilon
            denominator_slices = tf.gather(denominator, gradient.indices)
            increment = tf.IndexedSlices(
                lr * gradient.values * tf.math.rsqrt(denominator_slices),
                gradient.indices,
            )

            if self.momentum > 0:
                momentum.assign(self.momentum * momentum)
                momentum.scatter_add(increment)
                variable.assign_add(-momentum)
            else:
                variable.scatter_add(-increment)
        else:
            velocity.assign(rho * velocity + (1 - rho) * tf.square(gradient))
            if self.centered:
                average_grad.assign(rho * average_grad + (1 - rho) * gradient)
                denominator = velocity - tf.square(average_grad) + self.epsilon
            else:
                denominator = velocity + self.epsilon
            increment = lr * gradient * tf.math.rsqrt(denominator)
            if self.momentum > 0:
                momentum.assign(self.momentum * momentum + increment)
                variable.assign_add(-momentum)
            else:
                variable.assign_add(-increment)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
            }
        )
        return config
