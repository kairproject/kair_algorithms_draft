# -*- coding: utf-8 -*-
"""Simple test case(sine+noise->cos) for LSTM network."""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from algorithms.common.networks.lstm import LSTM


def noise_generator(size):
    return np.random.normal(0, 0.5, size=size)


def get_wave_func(wave_nm):
    assert wave_nm in ["sine", "cos"]

    if wave_nm == "sine":
        func = np.sin
    else:
        func = np.cos
    return func


def generate_wave_data(x_range, wave_nm="sine", noise=True, iters=10, steps=100):
    wave_data = []

    wave_func = get_wave_func(wave_nm)

    for _ in range(iters):
        if noise:
            wave_data.append(wave_func(x_range) + noise_generator(x_range.size))
        else:
            wave_data.append(wave_func(x_range))

    wave_data = np.expand_dims(np.array(wave_data), -1)
    return wave_data


def plot_overlapped_waves(x_range, input_wave, output_wave_nm):
    input_wave = input_wave.squeeze()
    iters, num_timesteps = input_wave.shape

    for i in range(iters):
        plt.plot(x_range, input_wave[i])

    output_wave_func = get_wave_func(output_wave_nm)
    plt.plot(x_range, output_wave_func(x_range), c="r", lw=10, alpha=0.3, label="True")

    plt.legend()
    plt.show()


class SineCosDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_timesteps, self.num_data, self.num_feat = x.shape

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        sample = {"x": x, "y": y, "idx": idx}
        return sample


if __name__ == "__main__":
    # hyperparameters
    x_range = np.linspace(-3 * np.pi, 3 * np.pi, 100)
    input_wave_nm = "sine"
    output_wave_nm = "cos"
    num_waves = 1000
    num_test_waves = 100
    num_timesteps = 100
    input_size = 1
    output_size = 1
    epochs = 100
    hidden_sizes = [128, 128, 128]

    # data
    input_wave = generate_wave_data(
        x_range, input_wave_nm, True, num_waves, num_timesteps
    )
    output_wave = generate_wave_data(
        x_range, output_wave_nm, False, num_waves, num_timesteps
    )
    sinecosdataset = SineCosDataset(input_wave, output_wave)
    dataloader = DataLoader(sinecosdataset, shuffle=True, batch_size=100)

    # model
    lstm = LSTM(
        input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes
    ).cuda()

    optimizer = optim.Adam(lstm.parameters())

    # train
    for i in range(epochs):
        for sample in dataloader:
            x = sample["x"].float().to("cuda")
            y = sample["y"].float().to("cuda")

            outputs = lstm(x)
            loss = F.mse_loss(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[epoch: {i}] loss: {loss}")

    # eval
    eval_data = generate_wave_data(x_range, input_wave_nm, True, num_test_waves)
    eval_data = torch.Tensor(eval_data).to("cuda")
    pred = lstm(eval_data)
    pred = pred.data.cpu().numpy().squeeze()

    # plot resulit
    plot_overlapped_waves(x_range, pred, output_wave_nm)
