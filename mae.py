import sys
import torch
import numpy as np
from PIL import Image

from mae_config import *
sys.path.append(MAE_DIR)
import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std  = np.array([0.229, 0.224, 0.225])

mask_len   = 196
mask_bytes = 25
default_mask_ratio = 0.75
default_checkpoint_name = 'mae_visualize_vit_large_ganloss.pth'

def mae_prepare_model(path_to_checkpoint, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()

    # load model
    checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
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

def mae_main(img_path: str, out_dir: str, mask_base64: str | None, checkpoint_name: str):
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
    mae_main(
        img_path        = sys.argv[1],
        out_dir         = sys.argv[2] if len(sys.argv) >= 3 else '.',
        mask_base64     = sys.argv[3] if len(sys.argv) >= 4 else None,
        checkpoint_name = sys.argv[4] if len(sys.argv) >= 5 else default_checkpoint_name)
