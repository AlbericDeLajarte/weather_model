import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import wandb
from tqdm import tqdm
import copy

import sys, os

import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

data_source = 'Pleiade'
base_path = '/Users/alberic/Desktop/divers/projects/hvac_opt/'
data_path = os.path.join(base_path, 'weather_model', 'processed_data', data_source)
with open(os.path.join(data_path, 'normalization.json')) as f:
    normalization_data = json.load(f)


i_start_out = 4
i_end_out = 5

class MLPModel(pl.LightningModule):

    def __init__(self, d_input: int, d_output: int, d_layers: list, #dataModule,
                 dropout: float = 0.0, lr: float = 1e-2, gamma: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        
        # Misc parameters
        self.model_type = 'MLP'
        self.lr = lr
        self.gamma = gamma
        self.train_losses = []
        self.valid_losses=[]

        # MLP layers
        self.input_layer = nn.Linear(d_input, d_layers[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(d_layers[i], d_layers[i+1]) for i in range(len(d_layers)-1)])
        self.output_layer = nn.Linear(d_layers[-1], d_output)

        self.activ = nn.ReLU()
        self.loss = nn.MSELoss()
        self.dropout = nn.Dropout(dropout)

        # self.dataModule = dataModule

        self.best_valid = np.inf

        save_path = os.path.join(base_path, 'weather_model', 'saved_models', data_source)
        new_idx = 0 if os.listdir(save_path)==[] else max([int(file.split('.pt')[0]) for file in os.listdir(save_path)]) + 1

        self.save_path = os.path.join(save_path, f'{new_idx}.pt')


    def forward(self, src: Tensor) -> Tensor:

        x = self.activ(self.input_layer(src))
        for layer in self.hidden_layers:
            x = self.dropout(self.activ(layer(x)))
        x = self.output_layer(x)

        return x


    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)

        # Compute loss and save it
        #loss = self.loss(outputs, targets) + 100*self.loss(inputs[:,-29+i_start_out:-29+i_end_out], targets)

        loss = self.loss(outputs, targets)
        self.train_losses.append(loss.item())
        # print(loss.item())
        self.log("train/loss", np.sqrt(loss.item()))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Next prediction loss
        outputs = self.forward(inputs)
        loss = np.sqrt(self.loss(outputs, targets).item())

        self.valid_losses.append(loss.item())

        if loss.item()<self.best_valid:
            self.best_valid = loss.item()
            torch.save(self.state_dict(), self.save_path)


        # Log loss
        # self.log("valid/future_loss", future_loss)
        self.log("valid/loss", loss)
        
        return loss


    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)

        return outputs

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='exp_range',  base_lr=1e-3, max_lr=2e-3, step_size_up=10, cycle_momentum=False, gamma=0.99)
        return [optimizer]#, [scheduler]
    

class Weather_dataset(torch.utils.data.Dataset):
    def __init__(self, datasets: int = 0, past_window: int = 1, data_type: str = 'train'):

        # Load dataset from library
        # datas = [np.load(os.path.join(base_path, 'weather_model', 'processed_data', 'singapore', file, 
        #                             f'normalized_data_{dataset}.npy')) for dataset in datasets]
        self.data = np.load(os.path.join(data_path, f'normalized_data_{data_type}.npy'))

        # self.input = np.concatenate([
        #                 np.concatenate( [data[k:len(data)-(past_window-k)] for k in range(past_window)], axis=1) 
        #         for data in datas])
        # self.target = np.concatenate([data[past_window:, i_start_out:i_end_out] for data in datas])

        self.input = np.concatenate( [self.data[k:len(self.data)-(past_window-k)] for k in range(past_window)], axis=1)
        self.target = self.data[past_window:, i_start_out:i_end_out]


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return ( torch.tensor(self.input[idx]).type(torch.float), 
                 torch.tensor(self.target[idx]).type(torch.float) )        
    

class Weather_dataModule(pl.LightningDataModule):

    def __init__(self, batch_train_size: int = 32, batch_valid_size: int = 32, past_window: int=1):
        super().__init__()
        self.save_hyperparameters()

        self.batch_train_size = batch_train_size
        self.batch_valid_size = batch_valid_size

        # self.train_dataset = Weather_dataset(datasets=[0,1,4,5,6,7], past_window=past_window)
        # self.valid_dataset = Weather_dataset(datasets=[2], past_window=past_window)
        # self.test_dataset = Weather_dataset(datasets=[3], past_window=past_window)

        self.train_dataset = Weather_dataset(past_window=past_window, data_type='train')
        self.valid_dataset = Weather_dataset(past_window=past_window, data_type='validation')
        self.test_dataset = Weather_dataset(past_window=past_window, data_type='test')

        self.idx = 0

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_train_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_valid_size, num_workers=0)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_valid_size, num_workers=0)
    
# Train loop
if __name__ == '__main__':

    wandb_logger = WandbLogger(project='weather_model')

    future_window = 32
    past_window = 16
    n_features = 19
    n_predic = (i_end_out-i_start_out)
    dataModule = Weather_dataModule(batch_train_size=128, batch_valid_size=128, past_window=past_window)
    
    ## To try:
    # - l2 norm
    # - batch/linear norm
    # - small batch size
    # - research on companies/commercial products

    model = MLPModel(d_input=n_features*past_window, d_output=n_predic,
                        d_layers=[256,128,64,32,16,8], dropout=0.0, lr=5e-4)

    max_epochs = 1000
    trainer = pl.Trainer(
                        accelerator='cpu', devices=1,
                        max_epochs=max_epochs,
                        log_every_n_steps=1,
                        accumulate_grad_batches=1,
                        enable_checkpointing=False,
                        logger=wandb_logger,
                        callbacks=  [
                                        LearningRateMonitor(logging_interval='epoch')
                                    ]
                                    )
    wandb_logger.watch(model, log='all', log_freq=100)
    trainer.fit(model=model, datamodule=dataModule)


    ## Testing loop ##

    # Create figure of all predictions
    model.dropout = nn.Dropout(0.0)
    model.load_state_dict(torch.load(model.save_path, map_location=torch.device('cpu')))
    test_data = dataModule.test_dataset.data
    
    time = test_data[:,0]*(12*31*24) + test_data[:,1]*(31*24) + test_data[:,2]*(24) + test_data[:,3]
    time -= time[0]
    
    def denormalize(output):

        output = copy.copy(output)

        # output[:,0] *= normalization_data['humidity'][1]
        # output[:,0] += normalization_data['humidity'][0]
        
        # output[:,1] *= normalization_data['temperature'][1]
        # output[:,1] += normalization_data['temperature'][0]

        output[:,0] *= normalization_data['temperature'][1]
        output[:,0] += normalization_data['temperature'][0]

        return output

    rmse_future = np.zeros(n_predic)

    fig = make_subplots(rows=n_predic, cols=1)
    
    test_data_denorm = denormalize(test_data[:, i_start_out:i_end_out])
    # fig.add_trace(go.Scatter(x=time, y=test_data_denorm[:,0], name='Humidity'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=time, y=test_data_denorm[:,1], name='Temperature'), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=test_data_denorm[:,0], name='Temperature'), row=1, col=1)


    for i in tqdm(range(0, len(test_data)-future_window-(past_window-1))):

        input = copy.copy(test_data[i:i+past_window].flatten())
        outputs = np.zeros((future_window, n_predic))
        for j in range(future_window):

            with torch.no_grad():
                outputs[j] = model(torch.tensor(input).unsqueeze(0).type(torch.float))[0]
                # print(torch.tensor(input).type(torch.float))

                if i+past_window+j<len(test_data):
                    input[:-n_features] = input[n_features:]
                    input[-n_features:] = copy.copy(test_data[i+j+past_window])
                    input[-n_features+i_start_out:-n_features+i_end_out] = outputs[j]

        rmse_future += np.mean(np.sqrt((denormalize(outputs) - test_data_denorm[i+past_window:i+past_window+future_window])**2), axis=0)

        if i%future_window==0:
            outputs = np.concatenate((test_data[i:i+past_window,i_start_out:i_end_out], outputs))
            outputs_denorm = denormalize(outputs)
            color = ['orange', 'green'][int((i/future_window)%2)]
            # fig.add_trace(go.Scatter(x=time[i:i+past_window+future_window], y=outputs_denorm[:, 0], opacity=0.5, marker_color=color, showlegend=False), row=1, col=1)
            # fig.add_trace(go.Scatter(x=time[i:i+past_window+future_window], y=outputs_denorm[:, 1], opacity=0.5, marker_color=color, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=time[i:i+past_window+future_window], y=outputs_denorm[:, 0], opacity=0.5, marker_color=color, showlegend=False), row=1, col=1)


    rmse_future /= len(test_data)-future_window


    # fig.update_layout(title=f'Temperature and humidity over time [hr], rmse over one hour: humidity={rmse_future[0]:.3f} and temperature={rmse_future[1]:.3f}',
    #                   height=1000)
    fig.update_layout(title=f'Temperature over time [hr], rmse over one hour: temperature={rmse_future[0]:.3f}',
                      height=1000)

    fig.show()