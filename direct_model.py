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
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, 'processed_data', data_source)
with open(os.path.join(data_path, 'normalization.json')) as f:
    normalization_data = json.load(f)


save_path = os.path.join(file_path, 'saved_models', 'direct', data_source)
new_idx = 0 if os.listdir(save_path)==[] else max([int(file.split('.pt')[0]) for file in os.listdir(save_path)]) + 1
save_path = os.path.join(save_path, f'{new_idx}.pt')

print(save_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

i_start_out = 4
i_end_out = 5

class TransformerModel(pl.LightningModule):

    def __init__(self, n_features_encoder: int, n_features_decoder: int, n_pred: int, d_model:int, n_heads: int, d_hid: int,
                 num_encoder_layers: int, num_decoder_layers:int, 
                 dropout: float = 0.1, lr:float = 1e-3, max_lr:float=1e-4, gamma:float=0.9):
        super().__init__()
        
        self.lr = lr
        self.max_lr = max_lr
        self.gamma = gamma

        self.n_pred = n_pred
        self.loss = nn.MSELoss()

        self.save_hyperparameters()
        
        self.model_type = 'Transformer'

        # self.transformer_encoder = nn.Sequential(*[TransformerEncoderLayer(d_model,nhead,d_hid,dropout,batch_first=True,norm_first=True,activation="gelu",)for i in range(nlayers)])
        self.transformer_model = nn.Transformer(d_model=d_model, dim_feedforward=d_hid, nhead=n_heads, 
                                                num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
                                                dropout=dropout, activation='gelu' ,batch_first=True, norm_first=True)
        
        self.encoder_adapter = nn.Linear(n_features_encoder, d_model)
        self.decoder_adapter = nn.Linear(n_features_decoder, d_model)
        self.decoder = nn.Linear(d_model, n_pred)
        self.decoder.bias.data.zero_()

        self.train_losses = []

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, D]

        Returns:
            output Tensor of shape [batch_size, seq_len, D]
        """
        input, target = src
        input = self.encoder_adapter(input)
        target = self.decoder_adapter(target)

        output = self.transformer_model(input, target)
        output = self.decoder(output)
        return output

    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)

        # Compute loss and save it
        loss = self.loss(outputs, targets)
        
        self.train_losses.append(loss.item())
        self.log("train/loss", np.sqrt(loss.item()))
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Next prediction loss
        outputs = self.forward(inputs)
        loss = np.sqrt(self.loss(outputs, targets).item())

        # self.valid_losses.append(loss.item())

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
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='exp_range',  base_lr=self.lr, max_lr=self.max_lr, step_size_up=10, cycle_momentum=False, gamma=self.gamma)
        return [optimizer], [scheduler]
    
class PredictionLogger(pl.Callback):

    def __init__(self, n_best=5, wandb_log=True):

        self.wandb_log = wandb_log
        
        # self.n_best=n_best
        self.best_valid = np.inf

        # self.test_examples = {'f1':[], 'pred':[], 'true':[], 'seq':[]}
        # self.valid_examples = {'pred':[], 'true':[], 'f1':[], 'seq':[]}
        # self.valid_pairing_sum = []

        self.best_score = 0.0

    def on_validation_end(self, trainer, pl_module):

        loss = trainer.logged_metrics['valid/loss']
        if loss<self.best_valid:
            self.best_valid = loss
            torch.save(pl_module.state_dict(), save_path)

        wandb.log({'valid/best_loss': self.best_valid})

class Weather_dataset(torch.utils.data.Dataset):
    def __init__(self, datasets: int = 0, past_window: int = 1, future_window:int=1, data_type: str = 'train'):

        # Load dataset from library
        # datas = [np.load(os.path.join(base_path, 'weather_model', 'processed_data', 'singapore', file, 
        #                             f'normalized_data_{dataset}.npy')) for dataset in datasets]
        self.data = np.load(os.path.join(data_path, f'normalized_data_{data_type}.npy'))
        self.past_window = past_window
        self.future_window = future_window

        # self.input = np.concatenate([
        #                 np.concatenate( [data[k:len(data)-(past_window-k)] for k in range(past_window)], axis=1) 
        #         for data in datas])
        # self.target = np.concatenate([data[past_window:, i_start_out:i_end_out] for data in datas])

        # self.input = np.concatenate( [self.data[k:len(self.data)-(past_window-k)] for k in range(past_window)], axis=1)
        # self.input = self.data[:-past_window]
        # self.target = self.data[past_window:]
        # self.target = self.data[past_window:, i_start_out:i_end_out]


    def __len__(self):
        return len(self.data)-self.past_window-self.future_window+1
    
    def __getitem__(self, idx):
        future_data = torch.tensor(self.data[idx+self.past_window:idx+self.past_window+self.future_window]).type(torch.float)

        return ( (torch.tensor(self.data[idx:idx+self.past_window]).type(torch.float), torch.concatenate((future_data[:, :i_start_out], future_data[:, i_end_out:]), dim=1) ),
                 future_data[:, i_start_out:i_end_out]   )       
    

class Weather_dataModule(pl.LightningDataModule):

    def __init__(self, batch_train_size: int = 32, batch_valid_size: int = 32, past_window: int=1, future_window:int=1):
        super().__init__()
        self.save_hyperparameters()

        self.batch_train_size = batch_train_size
        self.batch_valid_size = batch_valid_size

        # self.train_dataset = Weather_dataset(datasets=[0,1,4,5,6,7], past_window=past_window)
        # self.valid_dataset = Weather_dataset(datasets=[2], past_window=past_window)
        # self.test_dataset = Weather_dataset(datasets=[3], past_window=past_window)

        self.train_dataset = Weather_dataset(past_window=past_window, future_window=future_window, data_type='train')
        self.valid_dataset = Weather_dataset(past_window=past_window, future_window=future_window, data_type='validation')
        self.test_dataset = Weather_dataset(past_window=past_window, future_window=future_window, data_type='test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_train_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_valid_size, num_workers=0)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_valid_size, num_workers=0)
    

# Train loop
if __name__ == '__main__':

    use_wandb = 1

    wandb_logger = use_wandb and WandbLogger(project='weather_model')

    if len(wandb.config._as_dict())>1:
        batch_train_size = wandb.config.batch_train_size
        d_model = wandb.config.d_model
        n_heads = wandb.config.n_heads
        d_hid = wandb.config.d_hid
        n_features_encoder = wandb.config.n_features_encoder
        n_features_decoder = wandb.config.n_features_decoder
        num_encoder_layers = wandb.config.num_encoder_layers
        num_decoder_layers = wandb.config.num_decoder_layers
        dropout = wandb.config.dropout
        lr = wandb.config.lr
        max_lr = wandb.config.max_lr
        gamma = wandb.config.gamma
    else:
        batch_train_size = 128
        d_model = 32
        n_heads = 4
        d_hid = 32
        n_features_encoder = 19
        n_features_decoder = 18
        num_encoder_layers = 4
        num_decoder_layers = 4
        dropout = 0.3
        lr = 1e-4
        max_lr = 1e-3
        gamma = 0.9

    past_window = 64
    future_window = 32
    n_pred = (i_end_out-i_start_out)
    dataModule = Weather_dataModule(batch_train_size=batch_train_size, batch_valid_size=128, past_window=past_window, future_window=future_window)
    
    model = TransformerModel(   n_features_encoder=n_features_encoder, n_features_decoder=n_features_decoder, n_pred=n_pred, 
                                d_model = d_model, n_heads=n_heads, d_hid=d_hid, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                dropout = dropout, 
                                lr =lr, max_lr=max_lr, gamma=gamma)
    
    max_epochs = 20
    trainer = pl.Trainer(
                        accelerator=device, devices=1,
                        max_epochs=max_epochs,
                        log_every_n_steps=1,
                        accumulate_grad_batches=1,
                        enable_checkpointing=False,
                        logger=wandb_logger,
                        # precision="16-mixed",
                        callbacks=  [
                                        LearningRateMonitor(logging_interval='epoch'),
                                        PredictionLogger()
                                    ]
                                    )
    if use_wandb: wandb_logger.watch(model, log='all', log_freq=100)
    trainer.fit(model=model, datamodule=dataModule)


    ## Testing loop ##

    # Create figure of all predictions
    model = model.to(device)
    model.dropout = nn.Dropout(0.0)
    model.load_state_dict(torch.load(save_path, map_location=torch.device(device=device)))

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

    rmse_future = np.zeros(n_pred)

    fig = make_subplots(rows=n_pred, cols=1)

    test_data_denorm = denormalize(test_data[:, i_start_out:i_end_out])
    # fig.add_trace(go.Scatter(x=time, y=test_data_denorm[:,0], name='Humidity'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=time, y=test_data_denorm[:,1], name='Temperature'), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=test_data_denorm[:,0], name='Temperature'), row=1, col=1)


    for i in tqdm(range(0, len(dataModule.test_dataset))):

        input = test_data[i:i+past_window]
        target = test_data[i+past_window:i+past_window+future_window]
        target = np.concatenate((target[:, :i_start_out], target[:, i_end_out:]), axis=1)
        with torch.no_grad():
            outputs = model( (torch.tensor(input).to(device).unsqueeze(0).type(torch.float) ,torch.tensor(target).to(device).unsqueeze(0).type(torch.float)) )[0].cpu().numpy()

        rmse_future += np.mean(np.sqrt((denormalize(outputs) - test_data_denorm[i+past_window:i+past_window+future_window])**2), axis=0)

        if i%future_window==0:
            # outputs = np.concatenate((test_data[i:i+past_window,i_start_out:i_end_out], outputs))
            outputs_denorm = denormalize(outputs)
            color = ['orange', 'green'][int((i/past_window)%2)]
            # fig.add_trace(go.Scatter(x=time[i:i+past_window+future_window], y=outputs_denorm[:, 0], opacity=0.5, marker_color=color, showlegend=False), row=1, col=1)
            # fig.add_trace(go.Scatter(x=time[i:i+past_window+future_window], y=outputs_denorm[:, 1], opacity=0.5, marker_color=color, showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=time[i+past_window:i+past_window+future_window], y=outputs_denorm[:, 0], opacity=0.5, marker_color=color, showlegend=False), row=1, col=1)


    rmse_future /= len(test_data)-future_window


    # fig.update_layout(title=f'Temperature and humidity over time [hr], rmse over one hour: humidity={rmse_future[0]:.3f} and temperature={rmse_future[1]:.3f}',
    #                   height=1000)
    fig.update_layout(title=f'Temperature over time [hr], rmse over one hour: temperature={rmse_future[0]:.3f}',
                        height=1000)

    # fig.show()

    wandb.log({"final/trajectories": fig})