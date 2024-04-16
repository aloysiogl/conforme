# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

"""RNN model."""

import os.path
from typing import Optional, Tuple

import torch


def get_lengths_mask(sequences, lengths, horizon):
    """
    Returns the mask indicating which positions in the sequence are valid.

    Args:
        sequences: (batch of) input sequences
        lengths: the lengths of every sequence in the batch
        horizon: the forecasting horizon
    """

    lengths_mask = torch.zeros(sequences.size(0), horizon, sequences.size(2))
    for i, l in enumerate(lengths):
        lengths_mask[i, : min(l, horizon), :] = 1

    return lengths_mask


class RNN(torch.nn.Module):
    """
    RNN issuing point predictions.
    """

    def __init__(
        self,
        embedding_size: int,
        input_size: int = 1,
        output_size: int = 1,
        horizon: int = 1,
        rnn_mode: str = "LSTM",
        path: Optional[str] = None,
    ):
        """
        Args:
            embedding_size: hyperparameter indicating the size of the latent
                RNN embeddings.
            input_size: dimensionality of the input time-series
            output_size: dimensionality of the forecast
            horizon: forecasting horizon
            rnn_mode: type of the underlying RNN network
            path: optional path where to save the auxiliary model to be used
                in the main CFRNN network
        """
        super(RNN, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.horizon = horizon
        self.output_size = output_size
        self.path = path
        self.requires_fit = True

        self.rnn_mode = rnn_mode
        if self.rnn_mode == "RNN":
            self.forecaster_rnn = torch.nn.RNN(
                input_size=input_size, hidden_size=embedding_size, batch_first=True
            )
        elif self.rnn_mode == "GRU":
            self.forecaster_rnn = torch.nn.GRU(
                input_size=input_size, hidden_size=embedding_size, batch_first=True
            )
        else:  # self.mode == 'LSTM'
            self.forecaster_rnn = torch.nn.LSTM(
                input_size=input_size, hidden_size=embedding_size, batch_first=True
            )
        self.forecaster_out = torch.nn.Linear(embedding_size, horizon * output_size)

        if self.path and os.path.isfile(self.path):
            loaded_model = torch.load(self.path)
            self.load_state_dict(loaded_model.state_dict())
            for param in self.parameters():
                param.requires_grad = False
            self.requires_fit = False
            print("Loaded auxiliary forecaster from {}".format(path))
        else:
            print("Auxiliary forecaster not loaded.")

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        if state is not None:
            h_0, _ = state
        else:
            h_0 = None

        # [batch, horizon, output_size]
        if self.rnn_mode == "LSTM":
            _, (h_n, c_n) = self.forecaster_rnn(x.float(), state)
        else:
            _, h_n = self.forecaster_rnn(x.float(), h_0)
            c_n = None

        out = self.forecaster_out(h_n).reshape(-1, self.horizon, self.output_size)

        return out, (h_n, c_n)

    def fit(self, train_dataset, epochs: int, lr: float, batch_size: int = 32):
        """
        Trains the auxiliary forecaster to the training dataset.

        Args:
            train_dataset: a dataset of type `torch.utils.data.Dataset`
            batch_size: batch size
            epochs: number of training epochs
            lr: learning rate
        """
        if not self.requires_fit:
            return

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0

            for sequences, targets, lengths in train_loader:
                optimizer.zero_grad()

                out, _ = self(sequences)
                valid_out = out * get_lengths_mask(sequences, lengths, self.horizon)

                loss = criterion(valid_out.float(), targets.float())
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            if epoch % 50 == 0:
                print("Epoch: {}\tTrain loss: {}".format(epoch, mean_train_loss))

        if self.path is not None:
            torch.save(self, self.path)

    def get_point_predictions_and_errors(
        self, test_dataset: torch.utils.data.Dataset, corrected=True
    ):
        """
        Obtains point predictions of the examples in the test dataset.

        Obtained by running the Auxiliary forecaster and adding the
        calibrated uncertainty intervals.

        Args:
            test_dataset: test dataset
            corrected: whether to use Bonferroni-corrected calibration scores

        Returns:
            point predictions and their MAE compared to ground truth
        """
        self.eval()

        point_predictions = []
        errors = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for sequences, targets, lengths in test_loader:
            point_prediction, _ = self(sequences)
            point_predictions.append(point_prediction)
            errors.append(
                torch.nn.functional.l1_loss(
                    point_prediction, targets, reduction="none"
                ).squeeze()
            )

        point_predictions = torch.cat(point_predictions)
        errors = torch.cat(errors)

        return point_predictions, errors
