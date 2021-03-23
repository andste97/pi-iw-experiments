import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class Mnih2013(tf.keras.models.Model):
    def __init__(self, num_logits, dense_units=256, add_value=True, num_value_logits=1, use_batch_normalization=False, use_batch_renorm=False):
        super(Mnih2013, self).__init__()
        if use_batch_renorm:
            assert use_batch_normalization

        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=8,
                                            strides=4,
                                            padding="VALID",
                                            activation=None,
                                            name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=4,
                                            strides=2,
                                            padding="VALID",
                                            activation=None,
                                            name="conv2")
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=dense_units,
                                           activation=None,
                                           name="fc1")

        if use_batch_normalization:
            self.encoder = [self.conv1, tf.keras.layers.BatchNormalization(renorm=use_batch_renorm), tf.keras.layers.ReLU(),
                            self.conv2, tf.keras.layers.BatchNormalization(renorm=use_batch_renorm), tf.keras.layers.ReLU(),
                            self.flat, self.dense, tf.keras.layers.BatchNormalization(renorm=use_batch_renorm), tf.keras.layers.ReLU()]

        else:
            self.encoder = [self.conv1, tf.keras.layers.ReLU(), self.conv2, tf.keras.layers.ReLU(), self.flat, self.dense, tf.keras.layers.ReLU()]
            self.bn = None

        self.logits = tf.keras.layers.Dense(units=num_logits,
                                            activation=None,
                                            name="policy_logits")

        self.value = None
        if add_value:
            self.value = tf.keras.layers.Dense(units=num_value_logits,
                                               activation=None,
                                               name="value_logits")

    def call(self, x, training, **kwargs):
        output = {}
        for layer in self.encoder:
            x = layer(x, training=training)
            output[f"out_{layer.name}"] = x

        output["features"] = x
        output["policy_logits"] = self.logits(x, training=training)

        # Maybe compute value
        if self.value is not None:
            output["value_logits"] = self.value(x, training=training)

        return output


def my_tf_function(fn):
    graph_fn = tf.function(fn, autograph=False)
    def _fn(*args, **kwargs):
        new_args = [tf.constant(arg) for arg in args]
        new_kwargs = {k: tf.constant(v) for k,v in kwargs.items()}
        return graph_fn(*new_args, **new_kwargs)
    return _fn


def get_train_fn(model, optimizer, loss_fn, max_grad_norm=None, use_graph=False):
    def _train(*args, **kwargs):
        print("call train")
        with tf.GradientTape() as gtape:  # grads = self.optimizer.compute_gradients(batch)
            loss, res = loss_fn(*args, **kwargs)
        assert loss.shape == ()  # check it's a scalar!
        grads = gtape.gradient(loss, model.trainable_variables)
        if max_grad_norm is None:
            grad_norm = tf.linalg.global_norm(grads)
        else:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        res["global_gradients_norm"] = grad_norm
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, res

    if use_graph:
        return tf.function(_train, autograph=False)
    return _train


def get_loss_fn(model, args):
    def _loss(input_dict):
        output_dict = {}
        model_output = model(input_dict['observations'], training=True)

        cross_entropies = tf.nn.softmax_cross_entropy_with_logits(labels=input_dict['target_policy'],
                                                                  logits=model_output["policy_logits"])

        loss = tf.reduce_mean(cross_entropies, axis=0)
        output_dict['cross_entropy_loss'] = loss

        if args.compute_value:
            if args.use_value_classification:
                value_losses = tf.nn.softmax_cross_entropy_with_logits(labels=input_dict['returns'],
                                                                       logits=model_output["value_logits"])
                output_dict["value_logits"] = model_output["value_logits"]
            else:
                errors = input_dict['returns'] - tf.reshape(model_output["value_logits"], (-1,))
                output_dict['errors'] = errors
                value_losses = 0.5 * tf.square(errors)

            value_loss = tf.reduce_mean(value_losses, axis=0)
            output_dict['value_loss'] = value_loss

            loss = loss + args.value_factor * value_loss

        regularization = tf.reduce_sum([tf.nn.l2_loss(param) for param in model.variables], axis=0)
        output_dict["regularization_loss"] = regularization

        total_loss = loss + args.regularization_factor*regularization
        return total_loss, output_dict
    return _loss