"""
    define basic model
"""
def build_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10)
        ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

"""
    define transfer model
"""
def build_transfer_model(weights=[]):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


    # transfer layer freezing
    if len(weights) > 0:
        model.set_weights(weights)

    for i, l in enumerate(model.layers):
        if i <= 4:
            l.trainable = False
        else:
            l.trainable = True

    return model



"""
    federated learning start 
"""
max_round = 50
local_epochs = 20
fine_epochs = 10
client_num = 4

global_weight = []
for r in range(50): # number of round
    local_model_weight = []

    for i in range(4):  # number of client
        # local training
        model = build_model()

        if len(global_weight) > 0 :
            model.set_weights(global_weight)

        train_x, test_x, train_y, test_y = train_test_split(train_images, train_labels, test_size=0.3)

        test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)

        result = model.fit(train_x, train_y, epochs=local_epochs, validation_data=(test_x, test_y))

        local_model_weight.append(model.get_weights())

        # transfer learning
        agg_model = build_transfer_model(global_weight)
        # fine tuning
        agg_model.fit(train_x, train_y, epochs=fine_epochs)
        test_loss, test_acc = agg_model.evaluate(test_x, test_y, verbose=2)

        # validation
        other_loss, other_acc = agg_model.evaluate(test_images, test_labels)

    # weight avg
    global_weight = []
    for a, b, c, d in zip(local_model_weight[0], local_model_weight[1], local_model_weight[2], local_model_weight[3]):
        weight = (a + b + c + d) / 4
        global_weight.append(weight)