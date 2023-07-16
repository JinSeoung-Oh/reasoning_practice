import torch
from dalle2_pytorch import CLIP

clip = CLIP(
  dim_text = 512,
  dim_image = 512,
  dim_latent = 512,
  num_text_tokens=49408,
  text_enc_depth = 1,
  text_seq_len = 256,
  text_heads = 8,
  visual_enc_depth = 1,
  visual_image_size = 256,
  visual_patch_size=32,
  visual_heads=8,
  use_all_token_embeds=True,
  decoupled_contrastive_learning=True,
  use_visual_ssl = True,
  visual_ssl_type='simclr',
  use_mlm=False,
  text_ssl_loss_weight = 0.05,
  image_ssl_loss_weight=0.05).cuda()

# mock data
