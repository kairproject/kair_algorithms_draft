import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from algorithms.common.networks.lstm import LSTM


def noise_generator(size):
    return np.random.normal(0, 0.5, size=size)


def generate_wave_data(wave='sine', noise=True, iters=10, steps=100):
    assert wave in ['sine', 'cos']
    xs = np.linspace(-3*np.pi, 3*np.pi, 100)

    if wave == 'sine':
        func = np.sin
    else:
        func = np.cos
    wave_data = []

    for i in range(iters):
        wave_data.append(func(xs) + noise_generator(xs.size))

    wave_data = np.expand_dims(np.array(wave_data), -1)
    return wave_data


def plot_overlapped_waves(wave_data):
    wave_data = wave_data.squeeze()
    iters, num_timesteps = wave_data.shape

    for i in range(iters):
        plt.plot(range(num_timesteps), wave_data[i])

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

        sample = {'x': x, 'y': y, 'idx': idx}
        return sample


if __name__ == '__main__':
    sine_data = generate_wave_data('sine', True, 1000, 100)
    cos_data = generate_wave_data('cos', True, 1000, 100)
    sinecosdataset = SineCosDataset(sine_data, cos_data)
    dataloader = DataLoader(sinecosdataset, shuffle=True, batch_size=100)

    # hyperparameters
    input_size = 1
    output_size = 1
    epochs = 100
    hidden_sizes = [128, 128, 128]

    lstm = LSTM(input_size=input_size,
                output_size=output_size,
                hidden_sizes=hidden_sizes).cuda()

    optimizer = optim.Adam(lstm.parameters())
    sinecosdataset = SineCosDataset(sine_data, cos_data)
    dataloader = DataLoader(sinecosdataset, shuffle=True, batch_size=100)

    # train
    for i in range(epochs):
        for sample in dataloader:
            x = sample['x'].float().to('cuda')
            y = sample['y'].float().to('cuda')

            outputs = lstm(x)
            loss = F.mse_loss(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'[epoch: {i}] loss: {loss}')

    eval_data = generate_wave_data('sine', True, 100)
    eval_data = torch.Tensor(eval_data).to('cuda')
    pred = lstm(eval_data)
    pred = pred.data.cpu().numpy().squeeze()

    plot_overlapped_waves(pred)
