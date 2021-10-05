class SupervisedClassificationHyperModel:
    def model_builder(hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))
        model.add(keras.layers.Dense(10))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

    def create_tuner(self):
        tuner = kt.Hyperband(model_builder,
                             objective='val_accuracy',
                             max_epochs=10,
                             factor=3,
                             directory='my_dir',
                             project_name='intro_to_kt')

    def get_optimal_hyper_parameters(self):
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)


    def build_best_model(self):
        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        return model

    def get_best_hypermodel_epochs(self):
        history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        return best_epoch

    def train_hypermodel(self):
        best_hps = get_best_hypermodel_epochs()
        hypermodel = tuner.hypermodel.build(best_hps)

        # Retrain the model
        hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)


