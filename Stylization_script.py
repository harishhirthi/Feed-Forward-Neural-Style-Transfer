import os
import argparse 

import torch

from models.Transform_net import TransFormerNet
import utils as utils

def get_default_device() -> None:
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def stylization(Inference_config: dict):
    """
    Function to stylize the given content image.

    Args:
    Inference_config -> Dictionary of config params.
    
    """
    device = get_default_device()

    """Initializing Transformer model that stylizes the image"""
    style_net = TransFormerNet().to(device)
    trained_state = torch.load(os.path.join(Inference_config['binaries_path'], Inference_config['model_name']))
    binary = trained_state['state_dict']
    style_net.load_state_dict(binary, strict = True)
    style_net.eval()

    with torch.no_grad():
        content_img_path = os.path.join(Inference_config['content_img_path'], Inference_config['content_img_name'])
        content_img = utils.process_img(content_img_path, target_shape = Inference_config['target_shape'])
        content_img = content_img.to(device)
        stylized_img = style_net(content_img).detach().cpu().numpy().squeeze(0)
        utils.save_and_display(Inference_config, stylized_img, should_display = Inference_config['show_image'])
        print("Done stylizing.........")

if __name__ == '__main__':

    binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    img_save_path = os.path.join(os.path.dirname(__file__), 'output_images')

    os.makedirs(img_save_path, exist_ok = True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--content_img_dir", type = str, help = "Directory of content image.", default = "content") # Jumping-spider-C2-Macrotiff-PIxabay-PD.jpg
    parser.add_argument("--content_img_name", type = str, help = "Content image name.", default = "hilton-lagoon-hawaii.jpg") # O 1097 La.jpg, 00000014_(5).jpg, 00000005_(5).jpg
    parser.add_argument("--target_shape", type = int, help = "New width to be resized for content image", default = 700) # Jumping-spider-C2-Macrotiff-PIxabay-PD.jpg
    parser.add_argument("--model_name", type = str, help = "Model binary to be used for stylization", default = "style_styledc_cw_450000.0_sw_100000000000.0_tw_0.0_final.pth")
    parser.add_argument("--show_image", type = bool, help = "Show the output image.", default = False)
    args = parser.parse_args()

    inference_config = dict()
    for arg in vars(args):
        inference_config[arg] = getattr(args, arg)
    inference_config['content_img_path'] = os.path.join(os.path.dirname(__file__), args.content_img_dir)
    inference_config['save_folder'] = img_save_path
    inference_config['binaries_path'] = binaries_path

    stylization(inference_config)

