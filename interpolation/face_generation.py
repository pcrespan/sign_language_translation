import dnnlib
import legacy
import torch
import numpy as np
from PIL import Image
import os

def generate_progan_face(network_pkl='../models/nvidia/ffhq.pkl', output_path='../assets/synthetic_face.jpg', seed=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading: {network_pkl}')
    with open(network_pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    if seed is None:
        seed = np.random.randint(0, 2**32)
    print(f"Using seed: {seed}")
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    img = G(z, None, truncation_psi=0.7, noise_mode='const')
    img = (img.clamp(-1, 1) + 1) * (255 / 2)
    img = img.permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.uint8)

    Image.fromarray(img).save(output_path)
    print(f"Synthetic face saved at: {output_path}")