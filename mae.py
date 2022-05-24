import sys
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict

from mae_config import *
sys.path.append(MAE_DIR)
import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std  = np.array([0.229, 0.224, 0.225])

mask_len   = 196
mask_bytes = 25
default_mask_ratio = 0.75
default_checkpoint_name = 'mae_visualize_vit_large_ganloss.pth'
load_state_dict_message_success = "<All keys matched successfully>"
load_state_dict_message_no_missing_keys = "_IncompatibleKeys(missing_keys=[]"

def mae_fix_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith('mae.'):
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            name = key.replace("mae.", "") # remove `mae.`
            new_state_dict[name] = val
        return new_state_dict
    else:
        return state_dict

def mae_prepare_model(path_to_checkpoint, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()

    # load model
    checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    state_dict = mae_fix_state_dict(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    msg = str(msg)
    if msg == load_state_dict_message_success:
        print("okay.")
    else:
        if msg.startswith(load_state_dict_message_no_missing_keys):
            print("okay, some keys unexpected.")
        else:
            raise Exception(msg)
    return model

def mae_set_mask(model, mask, mask_ratio):
    def random_masking(x, _):
        N, L, D = x.shape # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # sort noise for each sample
        ids_shuffle = torch.argsort(mask, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask_res = torch.ones([N, L], device=x.device)
        mask_res[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask_res = torch.gather(mask_res, dim=1, index=ids_restore)

        return x_masked, mask_res, ids_restore
    model.random_masking = random_masking
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

def parse_mask(mask_base64: str):
    import base64
    buf = base64.b64decode(mask_base64)
    buf = np.frombuffer(buf, dtype=np.uint8)
    assert len(buf) == mask_bytes
    buf = np.unpackbits(buf)[:mask_len]

    mask_ratio = np.count_nonzero(buf) / mask_len

    mask = np.ndarray(shape=(1, mask_len))
    mask[0, :] = buf
    mask = torch.tensor(mask)
    return mask_ratio, mask

def mae_main(
    img_path        : str,
    out_dir         : str,
    mask_base64     : str | None,
    checkpoint_name : str | None):
    checkpoint_name = checkpoint_name or default_checkpoint_name

    print('loading model ... ', end='')
    model = mae_prepare_model(
        f'{MAE_CHECKPOINT_DIR}/{checkpoint_name}',
        f'mae_vit_large_patch16')
    print('model loaded.')

    img = Image.open(img_path)
    img = image_preproc(img)
    print('input image loaded.')

    if mask_base64 is not None:
        mask_ratio, mask = parse_mask(mask_base64)
        model = mae_set_mask(model, mask, mask_ratio)
        print(f'input mask loaded, mask_ratio: {mask_ratio}')
    else:
        mask_ratio = default_mask_ratio
        mask = torch.rand(1, mask_len, device=img.device)
        model = mae_set_mask(model, mask, mask_ratio)
        print(f'random mask generated, mask_ratio: {mask_ratio}')

    out = mae_core(img, model)
    print('reconstruction completed.')

    img = out['masked']
    img = image_postproc(img)
    img.save(f'{out_dir}/masked.png')

    img = out['recon']
    img = image_postproc(img)
    img.save(f'{out_dir}/recon.png')

    img = out['recon_visible']
    img = image_postproc(img)
    img.save(f'{out_dir}/recon_visible.png')

    print('output image saved.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
        dest = 'img_path'         , type = str, required = True,
        help = 'input image path')
    parser.add_argument('-o',
        dest = 'out_dir'          , type = str, required = True,
        help = 'output directory')
    parser.add_argument('-m',
        dest = 'mask_base64'      , type = str, default = None,
        help = 'input mask (base64)')
    parser.add_argument('-c',
        dest = 'checkpoint_name'  , type = str, default = None,
        help = 'checkpoint name')
    args = parser.parse_args()
    mae_main(args.img_path, args.out_dir, args.mask_base64, args.checkpoint_name)
