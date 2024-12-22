import logging
from scipy.stats import beta
import torch


logger = logging.getLogger(__name__)


def batch_preprocessing(batch, include_domain):
    return multibranch_division(batch)

def domain_on_all_or_none(batch, include_domain):
    x, y, rr_features, wavelet_features = batch
    x = torch.transpose(x, 1, 2)
    rr_features = torch.transpose(rr_features, 1, 2)
    wavelet_features = torch.transpose(wavelet_features, 1, 2)
    rr_x = torch.hstack((rr_features, x))
    rr_wavelets = torch.hstack((rr_features, wavelet_features))
    pre_pca = None
    if include_domain:
        pre_pca = torch.hstack((rr_features, x[:, ::2, :], wavelet_features))
    else:
        pre_pca = torch.hstack((x[:, ::2, :], wavelet_features))


    pca_features = torch.pca_lowrank(pre_pca)
    pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1],
                                    pca_features[2].reshape(pca_features[2].shape[0], -1)))
    pca_features = pca_features[:, :, None]

    if include_domain:
        alpha1_input = rr_x
        alpha2_input = rr_wavelets
        beta_input = pca_features
    else:
        alpha1_input = x
        alpha2_input = wavelet_features
        beta_input = pca_features

    logger.debug(f"Shape nf alpha1_input: {alpha1_input.shape}\nShape of alpha2_input: {alpha2_input.shape}\nBeta input shape: {beta_input.shape}\nPCA Features shape: {pca_features.shape}")
    return alpha1_input, alpha2_input, beta_input, y



def domain_at_beta_only(batch):
    x, y, rr_features, wavelet_features = batch
    x = torch.transpose(x, 1, 2)
    rr_features = torch.transpose(rr_features, 1, 2)
    wavelet_features = torch.transpose(wavelet_features, 1, 2)
    pre_pca = torch.hstack((x[:, ::2, :], wavelet_features))


    pca_features = torch.pca_lowrank(pre_pca)
    pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1],
                                    pca_features[2].reshape(pca_features[2].shape[0], -1)))
    logger.debug(f"PCA SHAPE after initial hstack: {pca_features.shape}")
    logger.debug(f"RR_features shape: {rr_features.shape}")
    rr_flat = torch.flatten(rr_features, start_dim=1)
    logger.debug(f"flattened shape: {rr_flat.shape}")
    pca_features = torch.hstack((pca_features, rr_flat))
    
    logger.debug(f"PCA SHAPE adter adding rr features: {pca_features.shape}")

    pca_features = pca_features[:, :, None]

    alpha1_input = x
    alpha2_input = wavelet_features
    beta_input = pca_features

    logger.debug(f"Shape nf alpha1_input: {alpha1_input.shape}\nShape of alpha2_input: {alpha2_input.shape}\nBeta input shape: {beta_input.shape}\nPCA Features shape: {pca_features.shape}")

    return alpha1_input, alpha2_input, beta_input, y


def domain_at_beta_no_pca(batch):
    x, y, rr_features, wavelet_features = batch
    x = torch.transpose(x, 1, 2)
    rr_features = torch.transpose(rr_features, 1, 2)
    wavelet_features = torch.transpose(wavelet_features, 1, 2)
    rr_flat = torch.flatten(rr_features, start_dim=1)
    logger.debug(f"flattened shape: {rr_flat.shape}")
    alpha1_input = x
    alpha2_input = wavelet_features
    beta_input = rr_flat[:, :, None]

    logger.debug(f"Shape nf alpha1_input: {alpha1_input.shape}\nShape of alpha2_input: {alpha2_input.shape}\nBeta input shape: {beta_input.shape}")

    return alpha1_input, alpha2_input, beta_input, y


def domain_at_mlp(batch):
    x, y, rr_features, wavelet_features = batch
    x = torch.transpose(x, 1, 2)
    rr_features = torch.transpose(rr_features, 1, 2)
    wavelet_features = torch.transpose(wavelet_features, 1, 2)
    pre_pca = torch.hstack((x[:, ::2, :], wavelet_features))


    pca_features = torch.pca_lowrank(pre_pca)
    pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1], pca_features[2].reshape(pca_features[2].shape[0], -1)))
    logger.debug(f"PCA SHAPE after initial hstack: {pca_features.shape}")
    logger.debug(f"RR_features shape: {rr_features.shape}")
    rr_flat = torch.flatten(rr_features, start_dim=1)
    logger.debug(f"flattened shape: {rr_flat.shape}")

    pca_features = pca_features[:, :, None]

    alpha1_input = x
    alpha2_input = wavelet_features
    beta_input = pca_features

    logger.debug(f"Shape nf alpha1_input: {alpha1_input.shape}\nShape of alpha2_input: {alpha2_input.shape}\nPCA Features shape: {pca_features.shape}")

    return alpha1_input, alpha2_input, beta_input, rr_flat, y


def multibranch_division(batch):
    x, y, rr_features, wavelet_features = batch
    x = torch.transpose(x, 1, 2)
    rr_features = torch.transpose(rr_features, 1, 2)
    wavelet_features = torch.transpose(wavelet_features, 1, 2)
    pre_pca = torch.hstack((x[:, ::2, :], wavelet_features, rr_features))


    pca_features = torch.pca_lowrank(pre_pca)
    pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1], pca_features[2].reshape(pca_features[2].shape[0], -1)))
    pca_features = pca_features[:, :, None].repeat(1,1,x.shape[2])

    alpha_input = x
    beta_input = wavelet_features
    gamma_input = pca_features
    delta_input = rr_features

    logger.debug(f"Shape nf alpha_input: {alpha_input.shape}\nShape of beta_input: {beta_input.shape}\nGamma shape: {gamma_input.shape}\nDelta input shape: {delta_input.shape}")

    return alpha_input, beta_input, gamma_input, delta_input , y
