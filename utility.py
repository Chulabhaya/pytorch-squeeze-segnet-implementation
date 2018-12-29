import torch.nn as nn
import torch
from torchvision import datasets, models, transforms

def init_vgg16_param():
    '''
    Returns the parameters of all 13 convolutional layers in VGG16 model.

    Inputs: 
    - None.

    Returns a tuple of:
    - layer_weights: List containing tensors for all conv layer weights.
    - layer_biases: List containing tensors for all conv layer biases.
    '''
    model = models.vgg16(pretrained=True)
    layer_weights = []
    layer_biases = []
    
    for module in model.features.children():
        if isinstance(module, nn.modules.conv.Conv2d):
            layer_weights.append(module.weight.data)
            layer_biases.append(module.bias.data)
            
    return layer_weights, layer_biases

def init_squeezenet_param(version=1.0):
    '''
    Returns the parameters of all convolutional layers in SqueezeNet model.

    Inputs: 
    - None.

    Returns a tuple of:
    - layer_weights: List containing tensors for all conv layer weights.
    - layer_biases: List containing tensors for all conv layer biases.
    '''
    if version == 1.0:
        model = models.squeezenet1_0(pretrained=True)
    else:
        model = models.squeezenet1_1(pretrained=True)

    layer_weights = []
    layer_biases = []
    
    for module in model.features.children():
        if isinstance(module, nn.modules.conv.Conv2d):
            layer_weights.append(module.weight.data)
            layer_biases.append(module.bias.data)
            
    for module in model.features.children():
        for module in module.children():
            if isinstance(module, nn.modules.conv.Conv2d):
                layer_weights.append(module.weight.data)
                layer_biases.append(module.bias.data)

    for module in model.classifier.children():
        if isinstance(module, nn.modules.conv.Conv2d):
            layer_weights.append(module.weight.data)
            layer_biases.append(module.bias.data)

    return layer_weights, layer_biases

def calculateIoU(prediction_batch, mask_batch):
    '''
    Returns mean Intersection over Union for batch of data. 
    Credit to: https://stackoverflow.com/a/42176782/6619979
    
    Inputs: 
    - prediction_batch: Tensor containing batch of predictions.
    - mask_batch: Tensor containing batch of masks. 

    Returns a tensor of:
    - mean_iou: Mean IoU of the batch. 
    '''    
    mean_iou = 0
    for i in range(len(prediction_batch)):
        y_pred = prediction_batch[i,:,:,:]
        y_true = mask_batch[i,:,:]
        y_pred = torch.argmax(y_pred, dim=0)
        y_pred = y_pred * (y_true)
        iteration_iou = (torch.sum((y_pred==y_true)*(y_true>0))).to(dtype=torch.float) / (torch.sum(y_true>0)).to(dtype=torch.float)
        mean_iou += iteration_iou
    mean_iou /= len(prediction_batch)
    return mean_iou
