import torch
import logging


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


    #TODO perform cleanup of train_network, validate_network and test_network to include batch preprocessing. Consider if batches loaded from data_loader should not be already preprocessed. 
    def batch_preprocessing(batch):
        x, y, rr_features, wavelet_features = batch
        x = torch.transpose(x, 1, 2)
        rr_features = torch.transpose(rr_features, 1, 2)
        wavelet_features = torch.transpose(wavelet_features, 1, 2)
        rr_x = torch.hstack((rr_features, x))
        rr_wavelets = torch.hstack((rr_features, wavelet_features))
        pre_pca = torch.hstack((rr_features, x[:, ::2, :], wavelet_features))
        pca_features = torch.pca_lowrank(pre_pca)
        pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1],
                                        pca_features[2].reshape(pca_features[2].shape[0], -1)))
        pca_features = pca_features[:, :, None]

        return x, y, rr_features, wavelet_features



    def train_network(self, model, training_data_loader, epoch):
            logger.info(f"...{epoch}/{self.training_config.num_epochs}")
            local_step = 0
            epoch_loss = []
            model.to(self.training_config.device)
            for x, y, rr_features, wavelet_features in training_data_loader:
                x = torch.transpose(x, 1, 2)
                rr_features = torch.transpose(rr_features, 1, 2)
                wavelet_features = torch.transpose(wavelet_features, 1, 2)
                rr_x = torch.hstack((rr_features, x))
                rr_wavelets = torch.hstack((rr_features, wavelet_features))
                pre_pca = torch.hstack((rr_features, x[:, ::2, :], wavelet_features))
                pca_features = torch.pca_lowrank(pre_pca)
                pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1],
                                                pca_features[2].reshape(pca_features[2].shape[0], -1)))
                pca_features = pca_features[:, :, None]
                local_step += 1
                model.train()
                forecast = model(rr_x.to(self.training_config.device), rr_wavelets.to(self.training_config.device), pca_features.to(self.training_config.device))
                #y_selected = torch.tensor(y.clone().detach(), self.training_config.device=self.training_config.device)
                loss = criterion(forecast, y.to(self.training_config.device))  # torch.zeros(size=(16,)))
                epoch_loss.append(loss)
                optimizer.zero_grad()
                loss.backward()
                if local_step % 50 == 0:
                    logger.info(f"Training loss at step {local_step} = {loss}")

            return epoch_loss




    def validate_network(self, model, validation_data_loader, epoch):
        logger.info(f"Entering validation, epoch: {epoch}")
        epoch_loss = []
        model.to(self.training_config.device)
        with torch.no_grad():
            model.eval()
            for x, y, rr_features, wavelet_features in validation_data_loader:
                x = torch.transpose(x, 1, 2)
                rr_features = torch.transpose(rr_features, 1, 2)
                wavelet_features = torch.transpose(wavelet_features, 1, 2)
                rr_x = torch.hstack((rr_features, x))
                rr_wavelets = torch.hstack((rr_features, wavelet_features))
                pre_pca = torch.hstack((rr_features, x[:, ::2, :], wavelet_features))
                pca_features = torch.pca_lowrank(pre_pca)
                pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1],
                                                pca_features[2].reshape(pca_features[2].shape[0], -1)))
                pca_features = pca_features[:, :, None]
                forecast = model(rr_x.to(self.training_config.device), rr_wavelets.to(self.training_config.device), pca_features.to(self.training_config.device))
                # , rr_wavelets.to(self.training_config.device), pca_features.to(self.training_config.device))
                #y_selected = torch.tensor(y.clone().detach(), self.training_config.device=self.training_config.device) # <- zmienione
                loss = criterion(forecast, y.to(self.training_config.device))
                epoch_loss.append(loss)
        return epoch_validation_loss


