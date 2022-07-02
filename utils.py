import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import cv2 as cv


def load_model_to_device(model):
    """
    Loads torch model to GPU if CUDA is available or to CPU otherwise.

    :param model: PyTorch model
    :type model: torch.nn.Module
    :return: model and used device
    :rtype: (torch.nn.Module, torch.device)
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print('\nCUDA is not available on your device or no compatible GPUs were found.\n'
              'Loading model to CPU...')
        device = torch.device('cpu')
    else:
        print('\nLoading model to GPU...')
        device = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    print(f'Successfully loaded the model to {"CPU" if device.type == "cpu" else "GPU"}!')

    return model, device


def torch_transform_image(img):
    """
    Transforms image for the PyTorch model.

    :param img: input image read with Pillow
    :type img: PIL.Image.Image
    :return: image transformed for the model
    :rtype: PIL.Image.Image
    """
    transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
    ])
    img = transform(img)
    return img


def blur(img):
    """
    Blurs image with gaussian blurring algorithm.

    :param img: input image
    :type img: numpy.ndarray
    :return: blurred image
    :rtype: numpy.ndarray
    """
    k_size = 11
    blurred = cv.GaussianBlur(img, (k_size, k_size), 0)
    return torch.FloatTensor(blurred)


def save_image(tensor, save_path,
               n_row=8, padding=2, pad_value=0, normalize=False, value_range=None, scale_each=False):
    """
    Saves PyTorch model prediction image to file

    :param tensor: output image tensor
    :type tensor: torch.tensor
    :param save_path: output file path
    :type save_path: str
    :param n_row: number of images displayed in each row of the output grid
    :type n_row: int
    :param padding: amount of padding
    :type padding: int
    :param pad_value: value put in the padded pixels
    :type pad_value: int
    :param normalize: if the output image should be normalized
    :type normalize: bool
    :param value_range: normalization min and max values,
    :type value_range: (int, int)
    :param scale_each: if every image in the batch should be scaled separately instead of the (min, max) over all images
    :type scale_each: bool
    :return: None
    """
    grid = utils.make_grid(
        tensor,
        nrow=n_row,
        padding=padding,
        pad_value=pad_value,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each
    )

    nd_arr = torch.round(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()
    nd_arr = nd_arr[:, :, 0]
    img = Image.fromarray(nd_arr)
    extension = save_path.split('.')[-1]
    if extension == "png":
        img.save(save_path)
    else:
        img.save(save_path, quality=100)  # for jpg
