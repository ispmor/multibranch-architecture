
import torch




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
    return x, y, rr_features, wavelet_features, rr_x, rr_wavelets, pca_features
