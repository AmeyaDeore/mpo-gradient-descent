import tensorflow as tf
from tqdm import tqdm
import math
import matplotlib.pyplot as plt


class MPOModel:

    def __init__(
        self,
        num_assets,
        loss_function,
        weights_function,
        get_best_weights_function,
        optimizer,
        random_weight_init=False,
    ):
        # Initialize portfolio weights as a TensorFlow trainable variable.
        if random_weight_init:
            self.z = tf.Variable(
                tf.random.normal(shape=(num_assets, 1), mean=1.0, stddev=0.01),
                trainable=True,
            )
        else:
            self.z = tf.Variable(tf.ones(shape=(num_assets, 1)), trainable=True)

        self.loss_function = loss_function
        self.weights_function = weights_function
        self.get_best_weights_function = get_best_weights_function
        self.optimizer = optimizer
        self.history = []

    def get_weights(self):
        return self.weights_function(self.z)

    def get_best_weights(self):
        return self.get_best_weights_function(self.history)

    def get_history(self):
        return self.history

    def _get_training_metrics_keys(self) -> list:
        """Gets training metrics.

        Returns:
            list: List of training metrics.
        """
        return list(self.history[0]["metrics"].keys())

    def plot_all_training_metrics(
        self,
        metrics: list = None,
        save_as: str = None,
        titles: dict = None,
    ):
        """
        Plots all specified training metrics.

        Parameters:
        metrics (list): List of metrics to plot. Defaults to None.
        split_lambdas_penalty (bool): Whether to plot lambdas and penalties separately. Defaults to False.
        save_as (str): File name to save the plot. Defaults to None.
        titles (dict): Dictionary with custom titles for metrics. Defaults to None.

        """
        self._plot_training_metrics_compact(
            history=[x["metrics"] for x in self.get_history()],
            metrics=(self._get_training_metrics_keys() if metrics is None else metrics),
            save_as=save_as,
            titles=titles,
        )

    def _plot_training_metrics_compact(
        self,
        history: list,
        metrics: list,
        save_as: str = None,
        titles: dict = None,
    ):
        """
        Plots specified training metrics compactly.

        Parameters:
        history (list): Training metrics history.
        metrics (list): List of metrics to plot.
        save_as (str): File name to save the plot.
        titles (dict): Dictionary with custom titles for metrics. Defaults to None.

        """
        metrics = [m for m in metrics if m.lower().startswith("loss")]
        n_metrics = len(metrics)
        n_cols = 4
        n_rows = math.ceil(n_metrics / n_cols)

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 3 * n_rows))

        if len(metrics) <= n_cols:
            for i, metric in enumerate(metrics):
                col = i % n_cols
                axs[col].plot([x[metric] for x in history])
                axs[col].set_title(titles[metric], fontdict={"fontsize": 15})
                axs[col].set_xlabel("Epoch", fontdict={"fontsize": 10})
                axs[col].set_ylabel(metric.capitalize(), fontdict={"fontsize": 10})
        else:
            for i, metric in enumerate(metrics):
                row = i // n_cols
                col = i % n_cols
                axs[row, col].plot([x[metric] for x in history])
                axs[row, col].set_title(titles[metric], fontdict={"fontsize": 15})
                axs[row, col].set_xlabel("Epoch", fontdict={"fontsize": 10})
                axs[row, col].set_ylabel(metric.capitalize(), fontdict={"fontsize": 10})

        # Remove empty subplots
        for j in range(i + 1, n_rows * n_cols):
            fig.delaxes(axs.flatten()[j])

        plt.tight_layout()

        if save_as is not None:
            plt.savefig(save_as)

        plt.show()

    def fit(self, x, idx=None, epochs=1):
        x = tf.constant(x, dtype=tf.float32)  # Asset returns
        if idx is not None:
            idx = tf.constant(idx, dtype=tf.float32)
        for e in tqdm(range(epochs)):
            with tf.GradientTape(persistent=True) as tape:
                w = self.get_weights()
                loss = self.loss_function(
                    assets_rets=x,
                    w=w,
                    idx=idx,
                )
            grads = tape.gradient(loss["loss"], [self.z])
            self.optimizer.apply_gradients(zip(grads, [self.z]))
            self.history.append(loss)
        return self.history
