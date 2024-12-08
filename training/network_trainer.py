from re import A
from networks.model import BlendMLP
from utilities import batch_preprocessing

import torch
import logging
import time

logger = logging.getLogger(__name__)


class TrainingConfig:
    batch_size = -1
    n_epochs_stop = -1
    num_epochs = -1
    lr_rate = 0.01
    criterion = None
    optimizer = None
    device=None
    def __init__(self, batch_size, n_epochs_stop, num_epochs, lr_rate, criterion, optimizer, device):
        self.batch_size=batch_size
        self.n_epochs_stop=n_epochs_stop
        self.num_epochs=num_epochs
        self.lr_rate=lr_rate
        self.criterion=criterion
        self.optimizer=optimizer
        self.device=device




class NetworkTrainer:
    min_val_loss = 999
    selected_classe = []
    training_config = None
    def __init__(self, selected_classes: list, training_config: TrainingConfig) -> None:
        self.selected_classe=selected_classes
        self.training_config = training_config
        logger.debug(f"Initiated NetworkTrainer object\n {self}")


    def train_network(self, model, training_data_loader, epoch, include_domain=True):
        logger.info(f"...{epoch}/{self.training_config.num_epochs}")
        local_step = 0
        epoch_loss = []
        model.to(self.training_config.device)
        for batch in training_data_loader:
            x, y, rr_features, wavelet_features, rr_x, rr_wavelets, pca_features = batch_preprocessing(batch, include_domain)
            local_step += 1
            model.train()
            if include_domain:
                forecast = model(rr_x.to(self.training_config.device), rr_wavelets.to(self.training_config.device), pca_features.to(self.training_config.device))
                logger.debug(f"Shape of x: {x.shape}\nShape of y: {y.shape}\nForecast shape: {forecast.shape}\nShape of rr_features: {rr_features.shape}\nWavelets feature shape: {wavelet_features.shape}\nPCA Features shape: {pca_features.shape}")
            else:
                forecast = model(x.to(self.training_config.device), wavelet_features.to(self.training_config.device), pca_features.to(self.training_config.device))
                logger.debug(f"Shape of x: {x.shape}\nShape of y: {y.shape}\nForecast shape: {forecast.shape}\nWavelets feature shape: {wavelet_features.shape}\nPCA Features shape: {pca_features.shape}")

            loss = self.training_config.criterion(forecast, y.to(self.training_config.device))  # torch.zeros(size=(16,)))
            epoch_loss.append(loss)
            self.training_config.optimizer.zero_grad()
            loss.backward()
            self.training_config.optimizer.step()
            if local_step % 50 == 0:
                logger.info(f"Training loss at step {local_step} = {loss}")

        logger.debug("Finished epoch training")
        result = torch.mean(torch.stack(epoch_loss))
        return result





    def validate_network(self, model, validation_data_loader, epoch, include_domain=True):
        logger.info(f"Entering validation, epoch: {epoch}")
        epoch_loss = []
        model.to(self.training_config.device)
        with torch.no_grad():
            model.eval()
            for batch in validation_data_loader:
                x, y, rr_features, wavelet_features, rr_x, rr_wavelets, pca_features = batch_preprocessing(batch, include_domain)

                if include_domain:
                    forecast = model(rr_x.to(self.training_config.device), rr_wavelets.to(self.training_config.device), pca_features.to(self.training_config.device))
                else:
                    forecast = model(x.to(self.training_config.device), wavelet_features.to(self.training_config.device), pca_features.to(self.training_config.device))

                loss = self.training_config.criterion(forecast, y.to(self.training_config.device))
                epoch_loss.append(loss)
        return torch.mean(torch.stack(epoch_loss))




    def train(self, blendModel, alpha_config, beta_config, training_data_loader, validation_data_loader, fold, leads, include_domain):
        best_model_name="default_model"
        epochs_no_improve=0
        min_val_loss=999999
        for epoch in range(self.training_config.num_epochs):
            epoch_loss = self.train_network(blendModel, training_data_loader, epoch, include_domain=include_domain)
            epoch_validation_loss = self.validate_network(blendModel, validation_data_loader, epoch, include_domain=include_domain)
            logger.info(f"Training loss for epoch {epoch} = {epoch_loss}")
            logger.info(f"Validation loss for epoch {epoch} = {epoch_validation_loss}")
            logger.info(f"not improving since: {epochs_no_improve}")
            if epoch_validation_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = epoch_validation_loss
                logger.info(f'Savining {len(leads)}-lead ECG model, epoch: {epoch}...')
                model_name = f"models_repository/{alpha_config.network_name}_{beta_config.network_name}_{leads}_{time.time()}.th"
                logger.debug(f"saving model: {model_name}")
                self.save(model_name,blendModel, self.training_config.optimizer, list(sorted(blendModel.classes)), leads)
                best_model_name=model_name
            else:
                epochs_no_improve += 1
            if epoch > 10 and epochs_no_improve >= self.training_config.n_epochs_stop:
                logger.warn(f'Early stopping!-->epoch: {epoch}; fold: {fold}')
                break
        return best_model_name




    def test_network():
        return 0




    def save(self, checkpoint_name, model, optimiser, classes, leads):
        torch.save({
            'classes': classes,
            'leads': leads,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict()
            }, checkpoint_name)



