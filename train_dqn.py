import tensorflow as tf

NUM_INPUTS = 4
NUM_OUTPUTS = 7

LEARNING_RATE = 0.001

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_object = tf.keras.losses.MeanSquaredError()

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_INPUTS,)),
        tf.keras.layers.Dense(24, activation=tf.nn.relu),
        tf.keras.layers.Dense(12, activation=tf.nn.relu),
        tf.keras.layers.Dense(NUM_OUTPUTS)])

def loss(model, x, y_true):
    y_pred = model(x)
    return loss_object(y_true=y_true, y_pred=y_pred)

#inputs = state, targets = true reward + future Q
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

q_network = build_model()
target_network = build_model()


loss_value, grads = grad(target_network, )

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, q_network.trainable_variables))