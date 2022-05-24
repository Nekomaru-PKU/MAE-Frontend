import sys
import torch
import numpy as np
from PIL import Image

from mae_config import *
sys.path.append(MAE_DIR)
import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std  = np.array([0.229, 0.224, 0.225])

def mae_prepare_model(path_to_checkpoint, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def mae_core(img: torch.Tensor, model):
    # make it a batch-like
    img = img.unsqueeze(dim=0)
    img = torch.einsum('nhwc->nchw', img)

    # run MAE
    loss, res, mask = model(img.float(), mask_ratio=0.15)
    res = model.unpatchify(res)
    res = torch.einsum('nchw->nhwc', res).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    img = torch.einsum('nchw->nhwc', img)

    return {
        'original': img[0],
        'masked': (img * (1 - mask))[0], # masked image
        'recon': res[0],

        # MAE reconstruction pasted with visible patches
        'recon_visible': (img * (1 - mask) + res * mask)[0],
    }

def image_preproc(img: Image) -> torch.Tensor:
    img = img.resize((224, 224)).convert('RGB')
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std
    img = torch.tensor(img)
    return img

def image_postproc(img: torch.Tensor) -> Image:
    img = torch.clip((img * imagenet_std + imagenet_mean) * 255, 0, 255)
    # print(img.shape)
    img = img.numpy()
    img = Image.fromarray(img.astype('uint8'))
    return img

default_checkpoint_name = 'mae_visualize_vit_large_ganloss.pth'

def mae_main(
    img_path       : str,
    out_dir        : str = '.',
    checkpoint_name: str = default_checkpoint_name):
    print('loading model...', end='')
    model_mae = mae_prepare_model(
        f'{MAE_CHECKPOINT_DIR}/{checkpoint_name}',
        f'mae_vit_large_patch16')
    print('model loaded.')

    img = Image.open(img_path)
    img = image_preproc(img)
    print('input image loaded.')

    torch.manual_seed(2)
    out = mae_core(img, model_mae)
    print('reconstruction completed.')

    img = out['recon_visible']
    img = image_postproc(img)
    img.save(f'{out_dir}/recon_visible.png')

    img = out['recon']
    img = image_postproc(img)
    img.save(f'{out_dir}/recon.png')

    print('output image saved.')

if __name__ == '__main__':
    mae_main(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) >= 3 else '.')
