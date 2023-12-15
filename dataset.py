from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torch.utils.data as data
from PIL import Image
from os import listdir
from os.path import join
import torch.nn.functional as F


def image_load(path):
    image = Image.open(path).convert('YCbCr')
    y, _, _ = image.split()
    return y
    

def cropRes_transform(image_size, scaling_factor):
    return Compose([
        CenterCrop(image_size),
        #Resize(image_size // scaling_factor),
        ToTensor(),
        lambda x : F.interpolate(x, size=(25, 25), mode="area"),
    ])
    

def crop_transform(image_size):
    return Compose([
        CenterCrop(image_size),
        ToTensor(),
    ])


def get_train_set(scaling_factor, directory, image_size):
    return DatasetFromFolder(directory,
                             input_transform=cropRes_transform(
                                 image_size, scaling_factor),
                             target_transform=crop_transform(image_size))


def get_val_set(scaling_factor, directory, image_size):
    return DatasetFromFolder(directory,
                             input_transform=cropRes_transform(
                                 image_size, scaling_factor),
                             target_transform=crop_transform(image_size))


def get_test_set(scaling_factor, directory, image_size):
    return DatasetFromFolder(directory,
                             input_transform=cropRes_transform(
                                 image_size, scaling_factor),
                             target_transform=crop_transform(image_size))


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform, target_transform):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x)
                                for x in listdir(image_dir)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = image_load(self.image_filenames[index])
        target = input.copy()
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)