import csv
from datetime import datetime

import numpy as np
import pypuf.batch
import pypuf.io
import pypuf.simulation.delay
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras


def count_occurrences(input_array):
    # Create a list to hold the counts of each element
    occurrence_counts = [0] * 21

    # Count the occurrences of each element in the input array
    for num in input_array:
        occurrence_counts[num] += 1

    # for num in occurrence_counts:
    #     occurrence_counts[num] = occurrence_counts[num]/100
    # occurrence_counts = occurrence_counts /101
    # occurrence_counts = np.divide(occurrence_counts,101)
    return occurrence_counts


class EarlyStopCallback(keras.callbacks.Callback):

    def __init__(self, acc_threshold, patience):
        super().__init__()
        self.accuracy_threshold = acc_threshold
        self.patience = patience
        self.default_patience = patience
        self.previous_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}

        # Stop the training when the validation accuracy reached the threshold accuracy
        if float(logs.get('val_accuracy')) > float(self.accuracy_threshold):
            print(f"\nReached {self.accuracy_threshold:2.2%}% accuracy, so stopping training!\n")
            self.model.stop_training = True

        # Stop the training when the validation acc is not enhancing for consecutive patience value
        if int(logs.get('val_accuracy')) < int(self.previous_accuracy):
            self.patience -= 1
            if not self.patience:
                print('\n*************************************************************************************')
                print('************** Break the training because of early stopping! *************************')
                print('*************************************************************************************\n')
                self.model.stop_training = True
        else:
            # Reset the patience value if the learning enhanced!
            self.patience = self.default_patience
        self.previous_accuracy = logs.get('accuracy')


# def run(k: int, n: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, weight1, weight2, weight3) -> dict:
def run(k: int, n: int, N: int, seed_sim: int, noisiness: float, BATCH_SIZE: int, weight1, weight2
            ) -> dict:
    patience = 5
    epochs = 200
    print('hello')
    challenges = pypuf.io.random_inputs(n=n, N=N, seed=seed_sim + 0)

    # XOR PUF
    puf = pypuf.simulation.delay.XORArbiterPUF(n=n, k=k, seed=seed_sim + 0, noisiness=noisiness)
    puf2 = pypuf.simulation.delay.XORArbiterPUF(n=n, k=k, seed=seed_sim + 0, noisiness=0)
    # puf = InterposePUF(n=n, k_up=1, k_down=k, seed=seed_sim + 0, noisiness=noisiness)
    # puf2 = InterposePUF(n=n, k_up=1, k_down=k, seed=seed_sim + 0, noisiness=0)
    puf_acc = 0
    eval_set_length = 100000
    val_chal = pypuf.io.random_inputs(n=n, N=eval_set_length, seed=seed_sim + 20)
    val_resp1 = puf.eval(val_chal)
    val_resp2 = puf2.eval(val_chal)
    for i in range(eval_set_length):
        if val_resp1[i] == val_resp2[i]:
            puf_acc += 1

    puf_acc = puf_acc / eval_set_length
    # FF XOR PUF
    # puf = XORFeedForwardArbiterPUF(n=n, k=k, ff=[(16, 32)], seed=seed_sim + 0, noisiness=noisiness)
    # puf2 = XORFeedForwardArbiterPUF(n=n, k=k, ff=[(16, 32)], seed=seed_sim + 0, noisiness=0)

    # IPUF
    # puf = InterposePUF(n=n, k_up=k, k_down=k, seed=seed_sim + 0, noisiness=noisiness)
    # puf2 = InterposePUF(n=n, k_up=k, k_down=k, seed=seed_sim + 0, noisiness=0)

    # for test dataset
    test_set_length = 10000
    test_challenges = pypuf.io.random_inputs(n=n, N=test_set_length, seed=seed_sim + 10)
    ground_truth = puf2.eval(test_challenges)
    test_challenges = np.cumprod(np.fliplr(test_challenges), axis=1, dtype=np.int8)

    # response 0 or 1
    # response2 response head
    responses2 = puf2.eval(challenges)
    for i in range(N):
        if responses2[i] == -1:
            responses2[i] = 0

    # response reliability head
    a_1 = puf.eval(challenges)
    a_1 = (a_1 + 1) / 2
    for i in range(19):
        tmp = puf.eval(challenges)
        tmp = (tmp + 1) / 2
        a_1 += tmp

    responses = np.array(a_1)
    responses = np.reshape(responses, (-1, 1))

    for i in range(49):
        tmp_responses = (puf.eval(challenges) + 1) / 2
        for j in range(19):
            tmp = puf.eval(challenges)
            tmp = (tmp + 1) / 2
            tmp_responses += tmp
        tmp_responses = np.array(tmp_responses)
        tmp_responses = np.reshape(tmp_responses, (-1, 1))
        responses = np.concatenate((responses, tmp_responses), axis=1)

    responses = responses.astype(int)

    res = [None] * N
    for i in range(N):
        tmp = count_occurrences(responses[i])
        tmp = np.divide(tmp, 50)
        res[i] = tmp

    res = np.array(res)
    responses = res


    challenges = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
    X_train, X_test, y_train, y_test, y_train2, y_test2 = train_test_split(challenges, responses,
                                                                                              responses2,
                                                                                              test_size=.1)
    # 3. setup early stopping
    # callbacks = EarlyStopCallback(0.98, patience)

    # 4. build network

    node = n
    inputs = tf.keras.Input(shape=(64,))
    s1 = 'relu'
    s2 = 'tanh'
    x1 = tf.keras.layers.Dense(64, activation=s1, kernel_initializer='random_normal')(inputs)
    x2 = tf.keras.layers.Dense(128, activation=s1, kernel_initializer='random_normal')(x1)
    x3 = tf.keras.layers.Dense(128, activation=s1, kernel_initializer='random_normal')(x2)

    x4 = tf.keras.layers.Dense(64, activation=s1, kernel_initializer='random_normal')(x3)
    x5 = tf.keras.layers.Dense(64, activation=s1, kernel_initializer='random_normal')(x4)

    x6 = tf.keras.layers.Dense(64, activation=s2, kernel_initializer='random_normal')(x3)
    x7 = tf.keras.layers.Dense(64, activation=s2, kernel_initializer='random_normal')(x6)

    outputs1 = tf.keras.layers.Dense(21, activation='softmax', name='new_sci')(x5)
    outputs2 = tf.keras.layers.Dense(1, activation='sigmoid', name='response')(x7)

    model = tf.keras.Model(inputs=inputs, outputs=[outputs1, outputs2])

    lossWeights = {"new_sci": weight1, "response": weight2}
    model.compile(
        loss={
            "new_sci": 'CategoricalCrossentropy',
            "response": 'binary_crossentropy',

        },

        metrics={
            "new_sci": 'accuracy',
            "response": 'accuracy',

        },

        optimizer='adam', loss_weights=lossWeights,
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    # 5. train
    started = datetime.now()

    history = model.fit(
        X_train, [y_train, y_train2], epochs=epochs, batch_size=BATCH_SIZE, callbacks=callback,
        shuffle=True, validation_split=0.01, verbose=0
    )
    # 6. evaluate result
    results = model.evaluate(X_test, [y_test, y_test2], batch_size=8, verbose=1)
    sets2, sets = model.predict(test_challenges)
    count = 0
    label = 0
    for i in range(test_set_length):
        if sets[i] <= 0.5:
            label = -1
        else:
            label = 1

        if label == ground_truth[i]:
            count += 1

    ACC = count / test_set_length

    fields = ['k', 'n', 'N', 'noise', 'training_size', 'test_accuracy', 'test_loss', 'time', 'seed', 'puf_acc',
              'sci_weight', 'response_weight',"sci2_weight"]

    # data rows of csv file
    # rows = [[k, n, N, noisiness, len(y_train), ACC, results[0], datetime.now() - started, seed_sim, puf_acc, weight1,
    #          weight2,weight3
    #          ]]
    rows = [[k, n, N, noisiness, len(y_train), ACC, results[0], datetime.now() - started, seed_sim, puf_acc, weight1,
             weight2
             ]]
    with open('./reliability_10xpuf_HARR_20x50_pypuf.csv', 'a') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        if seed_sim == 1:
            write.writerow(fields)
        write.writerows(rows)




def main(argv=None):
    # seed = [1, 16, 117, 344, 560, 687, 996, 1123, 2156, 3245]
    seed = [1, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    for i in seed:
        run(10, 64, 400000, i, 0.05, 1000, 1.8, 1)



if __name__ == '__main__':
    main()
