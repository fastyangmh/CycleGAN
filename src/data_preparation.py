# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.data_preparation import MyImageFolder, ImageLightningDataModule
from typing import Optional, Callable, TypeVar, Tuple, Any
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10
from glob import glob
from os.path import join
T_co = TypeVar('T_co', covariant=True)
import random
import numpy as np
from PIL import Image


#def
def create_datamodule(project_parameters):
    if project_parameters.predefined_dataset:
        dataset_class = {
            'MNIST': 'MNIST2CIFAR10',
            'CIFAR10': 'CIRFAR102MNIST'
        }[project_parameters.predefined_dataset]
        dataset_class = eval('My{}'.format(dataset_class))
    else:
        dataset_class = MyImageFolder
    return ImageLightningDataModule(
        root=project_parameters.root,
        predefined_dataset=project_parameters.predefined_dataset,
        classes=project_parameters.classes,
        max_samples=project_parameters.max_samples,
        batch_size=project_parameters.batch_size,
        num_workers=project_parameters.num_workers,
        device=project_parameters.device,
        transforms_config=project_parameters.transforms_config,
        target_transforms_config=project_parameters.target_transforms_config,
        dataset_class=dataset_class)


#class
class MyMNIST2CIFAR10(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        self.transform = transform
        self.classes = ['MNIST', 'CIFAR10']
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}
        self.mnist_data = MNIST(root=root,
                                train=train,
                                transform=None,
                                target_transform=None,
                                download=download)
        self.cifar10_data = CIFAR10(root=root,
                                    train=train,
                                    transform=None,
                                    target_transform=None,
                                    download=download)
        #balance data
        minimum_data_length = min(len(self.mnist_data), len(self.cifar10_data))
        self.mnist_data.data = self.mnist_data.data[:minimum_data_length]
        self.cifar10_data.data = self.cifar10_data.data[:minimum_data_length]
        self.minimum_data_length = minimum_data_length

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, index) -> T_co:
        sample1, _ = self.mnist_data[index]
        sample1 = sample1.convert('RGB')
        sample2, _ = self.cifar10_data[index]
        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        return sample1, sample2

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(self.minimum_data_length),
                                  k=max_samples)
            self.mnist_data = self.mnist_data[index]
            self.cifar10_data = self.cifar10_data[index]


class MyCIRFAR102MNIST(MyMNIST2CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.classes = ['CIFAR10', 'MNIST']
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}

    def __getitem__(self, index) -> T_co:
        sample2, sample1 = super().__getitem__(index)
        return sample1, sample2


class MyImageFolder(MyImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transform, target_transform)
        self.find_samples()

    def find_samples(self):
        samples = {}
        for c in self.classes:
            temp = []
            for ext in self.extensions:
                temp += glob(join(self.root, f'{c}/*{ext}'))
            samples[c] = sorted(temp)
        assert len(samples[self.classes[0]]) == len(
            samples[self.classes[1]]
        ), f'the {self.classes[0]} and {self.classes[1]} dataset have difference lengths.\nthe length of {self.classes[0]} dataset: {len(samples[self.classes[0]])}\nthe length of {self.classes[1]} dataset: {len(samples[self.classes[1]])}'
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples[self.classes[0]])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = []
        for c in self.classes:
            path = self.samples[c][index]
            s = self.loader(path)
            if self.transform:
                s = self.transform(s)
            sample.append(s)
        return sample[0], sample[1]

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(
                len(self.samples[self.classes[0]])),
                                  k=max_samples)
            self.samples = {
                k: np.array(v)[index]
                for k, v in self.samples.items()
            }


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample1 and sample2
    print('the dimension of sample1: {}'.format(x.shape))
    print('the dimension of sample2: {}'.format(y.shape))