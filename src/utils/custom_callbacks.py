import matplotlib.pyplot as plt
import keras.backend as K
import keras.callbacks
import os


class PrintLR(keras.callbacks.Callback):

    def __init__(self):
        super(keras.callbacks.Callback).__init__()

    def on_batch_end(self, batch, logs=None):
        print(' - LR:' + str(K.get_value(self.model.optimizer.lr)))


class SaveWeightCallback(keras.callbacks.Callback):
    """
    Callback, loads the weights from the latest epoch weights saved in the directory.
    When an epoch finishes, saves the weights into a new .weights file.
    """

    def __init__(self, save_folder):
        super(keras.callbacks.Callback).__init__()
        self._save_folder = save_folder
        self._epoch_count = 0

    def on_train_begin(self, logs=None):
        files = os.listdir(self._save_folder)

        files_numbers = []
        for f in files:
            try:
                f = f.replace('.weights', '')
                files_numbers.append(int(f))
            except ValueError:
                continue

        if len(files_numbers) == 0:
            return

        files_numbers.sort()

        self._epoch_count = files_numbers[-1]

        weights_path = os.path.join(self._save_folder, str(self._epoch_count)+'.weights')
        self.model.load_weights(weights_path)

        self._epoch_count += 1

    def on_epoch_end(self, epoch, logs=None):
        save_path = os.path.join(self._save_folder, str(self._epoch_count)+'.weights')
        self.model.save_weights()
        self._epoch_count += 1
