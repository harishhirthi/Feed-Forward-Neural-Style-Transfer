import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
 
# ImageNet Mean and Std values used for Normalization.
Imagenet_Mean = np.array([0.485, 0.456, 0.406])
Imagenet_Std = np.array([0.229, 0.224, 0.225])

def process_img(image_path: os.path, batch_size: int = 1, target_shape = None) -> torch.Tensor:
    """
    Function to Pre-process image.
    
    Args:
    image_path -> Path of image.
    batch_size -> To create batch of images.
    target_shape -> Target size of output image.

    """
    if not os.path.exists(image_path):
        raise Exception(f"Path doesn't exist: {image_path}")
    
    image = Image.open(image_path)
    # image = cv2.imread(image_path)[:, :, ::-1]
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            width, height = image.size
            # height, width = image.shape[:2]
            new_width = target_shape
            new_height = int(height * (new_width / width))
            image = image.resize((new_width, new_height))
            # image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
        else:
            image = image.resize((target_shape[0], target_shape[1]))
            # image = cv2.resize(image, (target_shape[0], target_shape[1]), interpolation = cv2.INTER_CUBIC)
    image = np.array(image).astype(np.float32)
    # image = image.astype(np.float32)
    image /= 255.0
    transforms = T.Compose([T.ToTensor(),
                            T.Normalize(mean = Imagenet_Mean, std = Imagenet_Std)
                           ])
    image = transforms(image)
    image = image.repeat(batch_size, 1, 1, 1)

    return image

"""________________________________________________________________________________________________________________________________________________________________"""

"""Class to create a subset of data to be used for training."""
class SequentialSampler(Sampler):
    
    def __init__(self, datasource, subset_size: int = None):
        """
        Args:
        datasource -> Torch Dataset.
        subset_size -> Number of samples to be used.
    
        """
        assert isinstance(datasource, Dataset) or isinstance(datasource, datasets.ImageFolder), "Datasource must be either Torch Dataset or Torchvision ImageFolder."
        self.data_source = datasource

        if subset_size is None:
            subset_size = len(self.data_source)

        assert 0 < subset_size <= len(self.data_source), f"Subset size must be between (0, {len(self.data_source)})."
        self.subset_size = subset_size
    
    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return self.subset_size

"""________________________________________________________________________________________________________________________________________________________________"""

def Train_DataLoader(training_config: dict) -> DataLoader:
    """
    Function to create DataLoader for training.

    Args:
    training_config -> Dictionary of params config.

    """
    transforms = T.Compose([T.Resize(training_config['image_size']),
                            T.CenterCrop(training_config['image_size']),
                            T.ToTensor(),
                            T.Normalize(mean = Imagenet_Mean, std = Imagenet_Std)
                           ])
    Train_dataset = datasets.ImageFolder(training_config['dataset_path'], transform = transforms)
    sampler = SequentialSampler(Train_dataset, subset_size = training_config['subset_size'])
    training_config['subset_size'] = len(sampler)
    train_dl = DataLoader(Train_dataset, batch_size = training_config['batch_size'], sampler = sampler, drop_last = True)
    return train_dl

"""________________________________________________________________________________________________________________________________________________________________"""

def gram_matrix(feature_map: torch.Tensor, should_mormalize: bool = True):
    """
    Function to create Gram matrix.

    Gram matrix - It captures the texture information from the feature maps extracted from convolutional
                  neural network. It is obtained using dot product between the feature maps. In other words,
                  it is a simple covariance matrix between different feature maps.
                  
    Args:
    feature_map -> Feature extraced from pre-trained vision models.
    should_normalize -> Should normalize the matrix.

    """
    (b, ch, h, w) = feature_map.size()
    features = feature_map.view(b, ch, h * w)
    features_t = features.transpose(1, 2)
    gram_mat = features.bmm(features_t)
    if should_mormalize:
        gram_mat = gram_mat / (ch * h * w)
    return gram_mat

"""________________________________________________________________________________________________________________________________________________________________"""

def total_variation_loss(image_batch) -> torch.Tensor:
    """
    Function for Total variation loss.
    Used for spatial continuity between the pixels of the generated image, thereby denoising it and giving it visual coherence.

    Args:
    image_batch -> Batch of image tensors.
    
    """
    batch_size = image_batch.shape[0]
    tv_height = torch.sum(torch.abs(image_batch[:, :, :-1, :] - image_batch[:, :, 1:, :]))
    tv_width = torch.sum(torch.abs(image_batch[:, :, :, :-1]) - image_batch[:, :, :, 1:])
    return (tv_height + tv_width) / batch_size

"""________________________________________________________________________________________________________________________________________________________________"""

def post_process_img(image: np.ndarray) -> np.ndarray:
    """
    Function to post process the generated image.
    
    Args:
    image -> Numpy array of generated image.

    """
    assert isinstance(image, np.ndarray), f"Expected Numpy Array, but got {type(image)}"
    mean = Imagenet_Mean.reshape(-1, 1, 1)
    std = Imagenet_Std.reshape(-1, 1, 1)
    image = (image * std) + mean
    image = (np.clip(image, 0., 1.) * 255).astype(np.uint8)
    image = np.moveaxis(image, 0, 2) 
    return image

"""________________________________________________________________________________________________________________________________________________________________"""

def save_and_display(config: dict, image: np.ndarray, should_display: bool = False, index: int = None):
    """
    Function to save and display images.

    Args:
    config -> Dictionary of config.
    image -> Numpy array of image.
    should_display -> To display image.
    index -> Index of generated image during training.

    """
    assert isinstance(image, np.ndarray), f"Expected Numpy Array, but got {type(image)}"
    image = post_process_img(image)

    if index is not None:
        fake_fname = 'generated-images-{0:0=4d}.jpg'.format(index)
    else:
        fake_fname = f'Stylized-image-{config["content_img_name"].split(".")[0]}.jpg'
    
    os.makedirs(config['save_folder'], exist_ok = True)
    plt.imsave(os.path.join(config['save_folder'], fake_fname), image)
    # cv2.imwrite(os.path.join(config['save_folder'], fake_fname), image[:, :, ::-1])

    if should_display:
        plt.imshow(image)
        plt.show()

"""________________________________________________________________________________________________________________________________________________________________"""

def print_parameters(model: torch.nn.Module) -> None:
    """
    Function to print number of parameters.
    
    Args:
    model -> Torch model

    """
    Num_of_parameters = sum(p.numel() for p in model.parameters())
    print("Model Parameters : {:.3f} M".format(Num_of_parameters / 1e6))