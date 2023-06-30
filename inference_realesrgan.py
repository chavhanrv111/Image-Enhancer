import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def main(**args):
    """Inference demo for Real-ESRGAN.
    """
    print(args)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']


    # determine model paths
    model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')

    # use dni to control the denoise strength
    dni_weight = None

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=not args['fp32'],
        gpu_id=None)

    if args['face_enhance'] == 'True':  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path=os.path.join('weights', 'GFPGANv1.3.pth'),
            upscale=args['outscale'],
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    # os.makedirs(args.output, exist_ok=True)

    path = args['input']

    # for idx, path in enumerate(paths):
    imgname, extension = os.path.splitext(os.path.basename(path))
    # print('Testing', idx, imgname)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    try:
        if args['face_enhance']:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=args['outscale'])
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        if args['ext'] == 'auto':
            extension = extension[1:]
        else:
            extension = args['ext']

        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        # if args.suffix == '':
        save_path = os.path.join(args['output'], f'{imgname}.{extension}')
        # else:
        #     save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
        cv2.imwrite(save_path, output)
        return save_path
            # return output

# if __name__ == '__main__':
#     main()
