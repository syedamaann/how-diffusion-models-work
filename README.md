# How Diffusion Models Work

In this post, I'll explain a Python implementation of a diffusion model using the PyTorch library, and I'll try to make it as detailed and understandable as possible.

### Introduction to Diffusion Models

Diffusion models are a class of generative models that progressively learn to reverse a process of adding random noise to data. Imagine starting with a clear image and gradually adding noise until it turns into a random cloud of pixels. Diffusion models learn to reverse this process to generate new images from noise, mimicking the data they were trained on.

### Setting Up the Environment

Before diving into the model, let's set up our environment:

```python
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from diffusion_utilities import *
```

We import necessary libraries such as `torch` for the model, `tqdm` for progress bars, and `matplotlib` for visualization. The `diffusion_utilities` might contain specific functions needed for our diffusion process.

### The Model: ContextUnet

The core of our implementation is the `ContextUnet` model, a U-Net-based neural network architecture:

```python
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):
        super(ContextUnet, self).__init__()
        ...
```

This model uses a U-Net architecture, which is effective for tasks where the output size is the same as the input size, such as image-to-image translation. It consists of a contracting path to capture context and a symmetric expanding path to enable precise localization.

![image](https://github.com/syedamaann/how-diffusion-models-work/assets/74735966/e93eaa60-0623-44bb-8d89-6bdb4bbcdf82)


#### Forward Pass

The forward method of the model defines how data flows through the network:

```python
def forward(self, x, t, c=None):
    x = self.init_conv(x)
    down1 = self.down1(x)
    down2 = self.down2(down1)
    hiddenvec = self.to_vec(down2)
    ...
    return out
```

`x` is the input image, `t` represents the time step, and `c` is an optional context label. The model first applies convolutions to downsample the image, embeds the time and context, then upsamples to construct the output from the embeddings and downsampled features.

### Training the Model

Training involves defining hyperparameters and setting up the noise schedule for the diffusion process:

```python
timesteps = 500
beta1 = 1e-4
beta2 = 0.02
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
...
nn_model = ContextUnet(...).to(device)
```

The `beta` values control the noise level at each step, crucial for the stability and quality of the generation.

### Fast Sampling with DDIM

Diffusion models can be sampled with different strategies. Here, we use DDIM, a faster variant:

```python
def denoise_ddim(x, t, t_prev, pred_noise):
    ...
    return x0_pred + dir_xt
```

This function adjusts the predicted clean image (`x0_pred`) based on the noise (`pred_noise`) estimated by the model. It's essential for generating clear images from noisy inputs.

### Visualization and Animation

Finally, we visualize the generated images and animate the generation process to see the model in action:

```python
samples, intermediate = sample_ddim(32, n=25)
animation_ddim = plot_sample(intermediate,32,4,save_dir, "ani_run", None, save=False)
HTML(animation_ddim.to_jshtml())
```

This block generates and displays images as they are being denoised, providing an intuitive understanding of the model's performance.

![image](https://github.com/syedamaann/how-diffusion-models-work/assets/74735966/d10f853d-47ae-4327-8243-be713027e94e)


### Conclusion

Diffusion models represent a powerful approach in generative AI, enabling a range of applications from image synthesis to more complex tasks like speech generation. With advancements in computing and algorithm design, they are becoming increasingly efficient and versatile.

**Acknowledgments:** This post was inspired by existing works and implementations such as those found at [minDiffusion](https://github.com/cloneofsimo/minDiffusion). The sprites and additional graphics used in the illustrations were provided by ElvGames, FrootsnVeggies, and kyrise.
