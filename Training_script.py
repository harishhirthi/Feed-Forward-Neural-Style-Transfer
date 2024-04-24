import os
import argparse 
import time
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from models.Perceptual_loss_net import Perceptual_Loss_Net
from models.Transform_net import TransFormerNet
import utils as utils


def get_default_device() -> torch.device:
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train(training_config: dict) -> None:
    """
    Function to train custom neural style transfer model.

    Args:
    training_config -> Dictionary of config params.
    
    """
    writer = SummaryWriter() # Tensorboard summary writer
    device = get_default_device() 

    train_dl = utils.Train_DataLoader(training_config) # Train Dataloader

    """Model initialization"""
    transform_net = TransFormerNet().to(device)
    perceptual_loss_net = Perceptual_Loss_Net().to(device)
    utils.print_parameters(transform_net)

    optimizer = torch.optim.Adam(transform_net.parameters(), lr = training_config['lr']) # Initializing Optimizer

    """To create style texture representation"""
    style_img_path = os.path.join(training_config['style_img_path'], training_config['style_img_name'])
    style_img = utils.process_img(image_path = style_img_path, batch_size = training_config['batch_size'])
    style_img = style_img.to(device)
    style_img_feature_maps_set = perceptual_loss_net(style_img) # Creating Feature maps for style image using vgg16
    target_style_representation = [utils.gram_matrix(x) for x in style_img_feature_maps_set] # Gram matrix for style feature maps

    pr_content_loss, pr_style_loss, pr_tv_loss = [0., 0., 0.]
    start_time = time.time()

    for epoch in range(training_config['epochs']):
        torch.cuda.empty_cache()
        stylized_img_batch = torch.empty((training_config["batch_size"], 3, training_config['image_size'], training_config['image_size']), dtype = torch.float32)
        for id, (content_img_batch, _) in enumerate(train_dl):
            optimizer.zero_grad()
            #print(content_img_batch.shape)

            content_img_batch = content_img_batch.to(device) # Original content image
            stylized_img_batch = transform_net(content_img_batch) # Transforming original content image to be stylized

            """Creating feature maps for original content and transformed content images using vgg16"""
            content_img_batch_feature_maps_set = perceptual_loss_net(content_img_batch)
            stylized_img_batch_feature_maps_set = perceptual_loss_net(stylized_img_batch)

            """Using ReLU2_2 feature maps to capture texture information from content image"""
            target_content_representation = content_img_batch_feature_maps_set.relu2_2
            current_content_representation = stylized_img_batch_feature_maps_set.relu2_2
            """Content loss: L2 distance between original content representation and transformed content representation"""
            content_loss = training_config['content_weight'] * torch.nn.MSELoss(reduction = 'mean')(target_content_representation, current_content_representation)
            
            style_loss = 0.0
            current_style_representation = [utils.gram_matrix(x) for x in stylized_img_batch_feature_maps_set] # Gram matrix for transformed content feature maps
            """Style loss: Sum of L2 distances between the Gram matrices of the representations of the content image and the style image"""
            for gram_tr, gram_con in zip(target_style_representation, current_style_representation):
                style_loss += torch.nn.MSELoss(reduction = 'mean')(gram_tr, gram_con)
            style_loss /= len(target_style_representation)
            style_loss *= training_config['style_weight']

            """Total variation loss"""
            tv_loss = training_config['tv_weight'] * utils.total_variation_loss(stylized_img_batch)

            total_loss = content_loss + style_loss + tv_loss # Total loss
            total_loss.backward()
            optimizer.step()

            pr_content_loss += content_loss.item()
            pr_style_loss += style_loss.item()
            pr_tv_loss += tv_loss.item()

            """Tensorboard logging"""
            if training_config['use_tensorboard']:
                writer.add_scalar("Loss/content_loss", content_loss.item(), len(train_dl) * epoch + id + 1)
                writer.add_scalar("Loss/style_loss", style_loss.item(), len(train_dl) * epoch + id + 1)
                writer.add_scalar("Loss/tv_loss", tv_loss.item(), len(train_dl) * epoch + id + 1)
                writer.add_scalars("Statistics/min-max-mean-median", {'min': torch.min(stylized_img_batch), 
                                  'max': torch.max(stylized_img_batch), 'mean': torch.mean(stylized_img_batch), 'median': torch.median(stylized_img_batch)},
                                  len(train_dl) * epoch + id + 1
                                  )
                if id % training_config['img_log_freq'] == 0:
                    stylized = utils.post_process_img(stylized_img_batch[0].detach().cpu().numpy())
                    stylized = np.moveaxis(stylized, 2, 0)
                    writer.add_image("Stylized_Image", stylized, len(train_dl) * epoch + id + 1)

        """Console logging"""
        if training_config['console_log_freq'] is not None and epoch % training_config['console_log_freq'] == 0:
            print(f'Time elapsed={(time.time() - start_time) / 60:.2f}(min) | epoch={epoch + 1} | c-loss={pr_content_loss / len(train_dl)} | s-loss={pr_style_loss / len(train_dl)} | tv-loss={pr_tv_loss / len(train_dl)} | total loss={(pr_content_loss + pr_style_loss + pr_tv_loss) / len(train_dl)}')
            pr_content_loss, pr_style_loss, pr_tv_loss = [0., 0., 0.]
            utils.save_and_display(training_config, stylized_img_batch[0].detach().cpu().numpy(), should_display = False, index = epoch)

        """Checkpoint model and optimizer states"""
        if training_config['checkpoint_freq'] is not None and epoch % training_config['checkpoint_freq'] == 0:
            training_state = dict()
            training_state['ckpt_state_dict'] = transform_net.state_dict()
            training_state['optimizer_state'] = optimizer.state_dict()
            ckpt_name = f'ckpt_style_{training_config["style_img_name"].split(".")[0]}_cw_{str(training_config["content_weight"])}_sw_{str(training_config["style_weight"])}_tw_{str(training_config["tv_weight"])}_epoch_{epoch}.pth'
            torch.save(training_state, os.path.join(training_config["checkpoint_path"], ckpt_name))

    """Saving Final model and optimizer states"""
    training_state = dict()
    training_state['state_dict'] = transform_net.state_dict()
    training_state['optimizer_state'] = optimizer.state_dict()
    model_name = f'style_{training_config["style_img_name"].split(".")[0]}_cw_{str(training_config["content_weight"])}_sw_{str(training_config["style_weight"])}_tw_{str(training_config["tv_weight"])}_final.pth'
    torch.save(training_state, os.path.join(training_config["binaries_path"], model_name))  


if __name__ == '__main__':

    binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    ckpt_root_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
    img_save_path = os.path.join(os.path.dirname(__file__), 'save_folder')
    batch_size = 4
    img_size = 256

    os.makedirs(binaries_path, exist_ok = True)
    os.makedirs(img_save_path, exist_ok = True)

    parser = argparse.ArgumentParser()

    # Please play with the weight parameters to check artistic morphing of content images.
    parser.add_argument("--dataset_root_directory", type = str, help = "Name of train images root directory.", default = "dataset")
    parser.add_argument("--style_img_root_directory", type = str, help = "Name of style images root directory.", default = "style")
    parser.add_argument("--style_img_name", type = str, help = "Name of style image used for training.", default = "styledc.jpg")
    parser.add_argument("--lr", type = float, help = "Learning rate for training.", default = 1e-3)
    parser.add_argument("--epochs", type = int, help = "Number of epochs for training.", default = 15)
    parser.add_argument("--content_weight", type = float, help = "Weight factor for content loss.", default = 4.5e5)
    parser.add_argument("--style_weight", type = float, help = "Weight factor for style loss.", default = 10e10)
    parser.add_argument("--tv_weight", type = float, help = "weight factor for total variation loss.", default = 0.0)
    parser.add_argument("--use_tensorboard", type = bool, help = "Use tensorboard logging.", default = True) # Run "tensorboard --logdir=runs --samples_per_plugin images=50" conda env
    parser.add_argument("--console_log_freq", type = int, help = "Display logs of training for epochs.", default = 1)
    parser.add_argument("--checkpoint_freq", type = int, help = "Checkpointing model and optimizer states for epochs.", default = 2)
    parser.add_argument("--img_log_freq", type = int, help = "Image logging in tensorboard.", default = 100)
    parser.add_argument("--subset_size", type = int, help = "Number of images to use for training.", default = None)
    args = parser.parse_args()

    ckpt_path = os.path.join(ckpt_root_path, args.style_img_name.split('.')[0])
    if args.checkpoint_freq is not None:
        os.makedirs(ckpt_path, exist_ok = True)

    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    training_config['dataset_path'] = os.path.join(os.path.dirname(__file__), args.dataset_root_directory)
    training_config['style_img_path'] = os.path.join(os.path.dirname(__file__), args.style_img_root_directory)
    training_config['binaries_path'] = binaries_path
    training_config['checkpoint_path'] = ckpt_path
    training_config['image_size'] = img_size
    training_config['batch_size'] = batch_size
    training_config['save_folder'] = img_save_path


    train(training_config)
