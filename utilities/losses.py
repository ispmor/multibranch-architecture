from sys import prefix
import torch
import logging

logger = logging.getLogger(__name__)

def challenge_metric_loss(predictions,true_labels, domain_weights):
    logger.debug(f"Predictions: {predictions}")
    logger.debug(f"true_labels: {true_labels}")

    normalizer=torch.sum(true_labels+predictions-true_labels*predictions,dim=1)
    normalizer[normalizer<1]=1
    logger.debug(f"Normalizer: {normalizer}")

    num_sigs,num_classes=list(true_labels.size())
    logger.debug(f"num_sigs: {num_sigs}, num_classes: {num_classes}")
    A=torch.zeros((num_classes,num_classes),dtype=true_labels.dtype)


    logger.debug(f"A: {A}")
    logger.debug(f"Domain weights: {domain_weights}")
    cuda_check = true_labels.is_cuda
    if cuda_check:
        cuda_device = true_labels.get_device()
        device = torch.device('cuda:' + str(cuda_device) )
        A=A.to(device)
        domain_weights=domain_weights.to(device)


    for sig_num in range(num_sigs):
        tmp=torch.matmul(torch.transpose(true_labels[[sig_num], :],0,1),predictions[[sig_num], :])/normalizer[sig_num]
        A=A + tmp
        logger.debug(f"A after addding tmp: {A}")
    cml = -torch.sum(A*domain_weights)
    logger.debug(f"Challenge metric loss= {cml}")
    return cml



def sparsity_loss(predictions):
    logger.debug(f"{predictions}")
    sl = torch.sum(-4*predictions * (predictions - 1))
    logger.debug(f"Sparsity loss: {sl}")
    return sl
