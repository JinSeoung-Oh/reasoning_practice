# git clone https://huggingface.co/Cene655/ImagenT5-3B
# pip install git+https://github.com/cene555/Imagen-pytorch.git
# pip install git+https://github.com/openai/CLIP.git
# git clone https://github.com/xinntao/Real-ESRGAN.git

# in Real-ESRGAN directory

# pip install basicsr
# pip install facexlin
# pip install gfpgan

# pip install -r requirement.txt
# python setup.py develop
# wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
# or go to link and download it

import PIL import Image
from Ipython.display import display #optional
import torch
from imagen_pytorch.model_creation import create_model_and_diffusion as create_model_and_diffusion_dalle2  #check github repository
from imagen_pytorch.model_creation import model_and_diffusion_defaults as model_and_diffusion_defaults_dalle2 #check github repository
from transformers import AutoTokenizer
import cv2

import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')


def model_fn(x_t, ts, **kwargs):
  guidance_scale = 5
  half = x_t[:len(x_t) // 2]
  combined = torch.cat([half, half], dim=0)
  model_out = model(combined, ts, **kwargs)
  eps, rest = model_out[:, :3], model_out[:, 3:]
  cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
  half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
  eps = torch.cat([half_eps, half_eps], dim=0)
  return torch.cat([eps, rest], dim=1)

def show_images(batch: th.Tensor):
    """ Display a batch of images inline."""
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))


def get_numpy_img(img):
    scaled = ((img + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([img.shape[2], -1, 3])
    return cv2.cvtColor(reshaped.numpy(), cv2.COLOR_BGR2RGB)


def _fix_path(path):
  d = th.load(path)
  checkpoint = {}
  for key in d.keys():
    checkpoint[key.replace('module.','')] = d[key]
  return checkpoint

options = model_and_diffusion_defaults_dalle2()
options['use_fp16'] = False
options['diffusion_steps'] = 200
options['num_res_blocks'] = 3
options['t5_name'] = 't5-3b'
options['cache_text_emb'] = True
model, diffusion = create_model_and_diffusion_dalle2(**options)

model.eval()

#if has_cuda:
#    model.convert_to_fp16()

model.to(device)


model.load_state_dict(_fix_path('/content/ImagenT5-3B/model.pt'))
print('total base parameters', sum(x.numel() for x in model.parameters()))

num_params = sum(param.numel() for param in model.parameters())

realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)

netscale = 4


upsampler = RealESRGANer(
    scale=netscale,
    model_path='/content/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth',
    model=realesrgan_model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True
)

face_enhancer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    upscale=4,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler
)

tokenizer = AutoTokenizer.from_pretrained(options['t5_name'])

prompt = 'A photo of cat'#@param {type:"string"}

text_encoding = tokenizer(
    prompt,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt"
)

uncond_text_encoding = tokenizer(
    '',
    max_length=128,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt"
)


import numpy as np
batch_size = 4
cond_tokens = th.from_numpy(np.array([text_encoding['input_ids'][0].numpy() for i in range(batch_size)]))
uncond_tokens = th.from_numpy(np.array([uncond_text_encoding['input_ids'][0].numpy() for i in range(batch_size)]))
cond_attention_mask = th.from_numpy(np.array([text_encoding['attention_mask'][0].numpy() for i in range(batch_size)]))
uncond_attention_mask = th.from_numpy(np.array([uncond_text_encoding['attention_mask'][0].numpy() for i in range(batch_size)]))
model_kwargs = {}
model_kwargs["tokens"] = th.cat((cond_tokens,
                                 uncond_tokens)).to(device)
model_kwargs["mask"] = th.cat((cond_attention_mask,
                               uncond_attention_mask)).to(device)

model.del_cache()
sample = diffusion.p_sample_loop(
    model_fn,
    (batch_size * 2, 3, 64, 64),
    clip_denoised=True,
    model_kwargs=model_kwargs,
    device='cuda',
    progress=True,
)[:batch_size]
model.del_cache()

show_images(sample)

for i in sample:
    show_images(i.unsqueeze(0))

new_img = get_numpy_img(sample)

for j in range(batch_size):
    new_img = get_numpy_img(sample[j].unsqueeze(0))
    for i in range(1):
        _, _, new_img = face_enhancer.enhance(new_img, has_aligned=False,
                                              only_center_face=False, paste_back=True)
        cv2.imwrite(f'/content/test_out{j}.jpg', new_img)



