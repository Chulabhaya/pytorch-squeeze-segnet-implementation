from squeeze_segnet import SqueezeSegNet, SqueezeSegNetOptimized
import numpy as np
import torch

encoder = SqueezeSegNetOptimized(3, 11)

img = torch.randn([4, 3, 480, 360])

output_encoder = encoder(img)

print(output_encoder.shape)