import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import wandb

import sys, os

class MLPModel(pl.LightningModule):

    def __init__(self, d_input: int, d_output: int, d_layers: list, dataModule,
                 dropout: float = 0.0, lr: float = 1e-2, gamma: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        
        # Misc parameters
        self.model_type = 'MLP'
        self.lr = lr
        self.gamma = gamma
        self.train_losses = []

        # MLP layers
        self.input_layer = nn.Linear(d_input, d_layers[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(d_layers[i], d_layers[i+1]) for i in range(len(d_layers)-1)])
        self.output_layer = nn.Linear(d_layers[-1], d_output)

        self.activ = nn.ReLU()
        self.loss = nn.MSELoss()
        self.dropout = nn.Dropout(dropout)

        self.dataModule = dataModule


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
        loss = self.loss(outputs, targets)
        # print(loss.item())
        self.log("train/loss", np.sqrt(loss.item()))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Cumulative loss
        # future_loss = 0 
        # for input in inputs:

        #     idx = self.dataModule.idx
        #     for i in range(self.dataModule.future_window):

        #         if idx+i < len(self.dataModule.valid_dataset.input)-1:

        #             next_state = torch.tensor(self.dataModule.valid_dataset.input[idx+i+1]).type(torch.float)
        #             output = self.forward(input)
        #             future_loss += np.sqrt(self.loss(output, next_state[6:8]).item())

        #             input = next_state
        #             input[6:8] = output
            
        #     # Increment idx as we are passing through valid in order
        #     if idx == len(self.dataModule.valid_dataset.input)-1:
        #         self.dataModule.idx = 0
        #     else:
        #         self.dataModule.idx +=1
                
        # future_loss /= targets.shape[0]*self.dataModule.future_window

        

        # Next prediction loss
        outputs = self.forward(inputs)
        loss = np.sqrt(self.loss(outputs, targets).item())

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='exp_range',  base_lr=1e-3, max_lr=5e-3, step_size_up=10, cycle_momentum=False, gamma=0.99)
        return [optimizer], [scheduler]
    

class Weather_dataset(torch.utils.data.Dataset):
    def __init__(self, datasets: int = 0, future_window: int = 1):

        # Load dataset from library
        datas = [np.load(os.path.join('/Users/alberic/Desktop/divers/projects/hvac_opt/weather_model/data/combined_Room1', 
                                    f'normalized_data_{dataset}.npy')) for dataset in datasets]
        
        # Try making it continous !! Input and target don't match like this
        np.random.seed(42)
        datas[0] = datas[0]

        self.input = np.concatenate([data[:len(data)-1] for data in datas])
        self.target = np.concatenate([data[1:, 6:8] for data in datas])
        self.future_window = future_window

        print(self.input)
        print(self.target)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return ( torch.tensor(self.input[idx]).type(torch.float), 
                 torch.tensor(self.target[idx]).type(torch.float) )        
    

class Weather_dataModule(pl.LightningDataModule):

    def __init__(self, batch_train_size: int = 32, batch_valid_size: int = 32, future_window=1):
        super().__init__()
        self.save_hyperparameters()

        self.batch_train_size = batch_train_size
        self.batch_valid_size = batch_valid_size

        # self.train_dataset = Weather_dataset(datasets=[1], future_window=future_window)
        self.train_dataset = Weather_dataset(datasets=[0,1,2,5,6,7], future_window=future_window)
        self.valid_dataset = Weather_dataset(datasets=[1], future_window=future_window)
        self.test_dataset = Weather_dataset(datasets=[4], future_window=future_window)

        self.idx = 0
        self.future_window = future_window

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_train_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_valid_size, num_workers=0)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_valid_size, num_workers=0)
    
# Train loop

if __name__ == '__main__':

    wandb_logger = WandbLogger(project='weather_model')

    future_window = 64
    dataModule = Weather_dataModule(batch_train_size=64, batch_valid_size=32, future_window=future_window)
    
    model = MLPModel(d_input=29, d_output=2, d_layers=[128, 64, 32, 16, 8], dataModule=dataModule)


    trainer = pl.Trainer(
                        accelerator='cpu', devices=1,
                        max_epochs=500,
                        log_every_n_steps=1,
                        accumulate_grad_batches=1,
                        enable_checkpointing=False,
                        logger=wandb_logger,
                        callbacks=[
                                        LearningRateMonitor(logging_interval='epoch')
                        #             EarlyStopping(monitor="valid/loss", mode='min', patience=30),
                        #             PredictionLogger(),
                        #             ModelChecker(log_every_nstep=100000)
                        ]
                                    )

    trainer.fit(model=model, datamodule=dataModule)


    # Create figure of all predictions
    test_data = np.load(os.path.join('/Users/alberic/Desktop/divers/projects/hvac_opt/weather_model/data/combined_Room1', 
                                    f'normalized_data_{1}.npy')) 
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # input = dataModule.train_dataset.input
    # target = dataModule.train_dataset.target

    # fig = make_subplots(rows=2, cols=2, shared_xaxes=True)
    # fig.add_trace(go.Bar(x=[str(np.round(inp, 3)) for inp in input], y=target[:,0]), row=1, col=1)
    # fig.add_trace(go.Bar(x=[str(np.round(inp, 3)) for inp in input], y=target[:,1]), row=2, col=1)

    # prediction = model(torch.tensor(input).type(torch.float))

    # fig.add_trace(go.Scatter(x=[str(np.round(inp, 3)) for inp in input], y=prediction[:,0].detach()), row=1, col=1)
    # fig.add_trace(go.Scatter(x=[str(np.round(inp, 3)) for inp in input], y=prediction[:,1].detach()), row=2, col=1) 

    # fig.update_layout(width=1500, height=4000)
    # fig.show()

    fig = make_subplots(rows=2, cols=1)
    time = test_data[:,0]*(12*31*24*60) + test_data[:,1]*(31*24*60) + test_data[:,2]*(24*60) + test_data[:,3]*(60)
    fig.add_trace(go.Scatter(x=time, y=test_data[:, 6]), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=test_data[:, 7]), row=2, col=1)


    for i in range(0, len(test_data), future_window):

        input = test_data[i]
        outputs = np.zeros((future_window//2+1, 2))
        outputs[0] = input[6:8]
        for j in range(future_window//2):

            with torch.no_grad():
                outputs[j+1] = model(torch.tensor(input).unsqueeze(0).type(torch.float))[0]
                # print(torch.tensor(input).type(torch.float))

                if i+1+j<len(test_data):
                    input = test_data[i+1+j]
                    input[6:8] = outputs[j+1]

        fig.add_trace(go.Scatter(x=time[i:i+future_window+1], y=outputs[:, 0], opacity=0.5, marker_color='orange', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=time[i:i+future_window+1], y=outputs[:, 1], opacity=0.5, marker_color='orange', showlegend=False), row=2, col=1)

    fig.show()