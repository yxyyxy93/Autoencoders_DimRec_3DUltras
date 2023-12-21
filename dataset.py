# Copyright
# Xiaoyu (Leo) Yang
# Nanyang Technological University
# 2023
# ==============================================================================
import queue
import threading

import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from utils import imgproc
from utils.Read_CSV import read_csv_to_3d_array

__all__ = [
    "TrainValidImageDataset",
    "PrefetchGenerator",
    "PrefetchDataLoader",
    "CPUPrefetcher", "CUDAPrefetcher",
]


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        label_dir (str): Directory where the corresponding labels are stored.
        mode (str): Data set loading method, the training data set is for data enhancement, and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, label_dir: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode
        # Get all subdirectories in the image directory
        self.subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        # Create a mapping from dataset files to label files
        self.dataset_label_mapping = self._create_dataset_label_mapping()

    def _create_dataset_label_mapping(self):
        mapping = {}
        # Iterate through each subdirectory
        for subdir in self.subdirs:
            dataset_path = os.path.join(self.image_dir, subdir)
            label_path = os.path.join(self.label_dir, subdir)
            # Get all dataset and label files in the subdirectory
            dataset_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            label_file = [f for f in os.listdir(label_path) if f.startswith('structure')]
            # Map each dataset file to its corresponding label file
            for dataset_file in dataset_files:
                # Assuming the file names are the same except for the prefix 'structure'
                # and the extension '.00.csv'
                full_dataset_file = os.path.join(dataset_path, dataset_file)
                full_label_file = os.path.join(label_path, label_file[0])
                mapping[full_dataset_file] = full_label_file

        return mapping

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Use the mapping to get the dataset and label files
        dataset_file = list(self.dataset_label_mapping.keys())[batch_index]
        label_file = self.dataset_label_mapping[dataset_file]

        # Load the images
        image_noisy = read_csv_to_3d_array(dataset_file)
        image_origin = read_csv_to_3d_array(label_file)

        # change dimension
        image_noisy = image_noisy.transpose(2, 1, 0)
        image_origin = image_origin.transpose(2, 0, 1)

        image_noisy = imgproc.normalize(image_noisy)

        # Convert to PyTorch tensors
        origin_tensor = torch.from_numpy(image_origin).long()
        noisy_tensor = torch.from_numpy(image_noisy).float()

        return {"gt": origin_tensor, "lr": noisy_tensor}

    def __len__(self) -> int:
        return len(self.dataset_label_mapping)


class TestDataset(Dataset):
    """
    Define test dataset loading methods.
    Args:
        image_dir (str): Test dataset directory.
        label_dir (str): Directory where the corresponding labels are stored.
    """

    def __init__(self, image_dir: str, label_dir: str) -> None:
        super(TestDataset, self).__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        # Get all subdirectories in the image directory
        self.subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        self.dataset_label_mapping = self._create_dataset_label_mapping()

    def _create_dataset_label_mapping(self):
        mapping = {}
        # Iterate through each subdirectory
        for subdir in self.subdirs:
            dataset_path = os.path.join(self.image_dir, subdir)
            label_path = os.path.join(self.label_dir, subdir)
            # Get all dataset and label files in the subdirectory
            dataset_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            label_files = [f for f in os.listdir(label_path) if f.startswith('structure')]
            # Map each dataset file to its corresponding label file
            for dataset_file in dataset_files:
                full_dataset_file = os.path.join(dataset_path, dataset_file)
                full_label_file = os.path.join(label_path, label_files[0])  # Adjust as needed
                mapping[full_dataset_file] = full_label_file
        return mapping

    def __getitem__(self, index: int) -> [torch.Tensor, torch.Tensor]:
        dataset_file = list(self.dataset_label_mapping.keys())[index]
        label_file = self.dataset_label_mapping[dataset_file]
        # Load the images
        image_noisy = read_csv_to_3d_array(dataset_file)
        image_origin = read_csv_to_3d_array(label_file)

        # change dimension
        image_noisy = image_noisy.transpose(2, 1, 0)
        image_origin = image_origin.transpose(2, 0, 1)

        image_noisy = imgproc.normalize(image_noisy)

        # Convert to PyTorch tensors
        origin_tensor = torch.from_numpy(image_origin).long()
        noisy_tensor = torch.from_numpy(image_noisy).float()

        return {"gt": origin_tensor, "lr": noisy_tensor}

    def __len__(self) -> int:
        return len(self.dataset_label_mapping)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        # Using 'next' on the iterator, and catch the StopIteration exception
        try:
            data = next(self.dataloader_iter)
            return data
        except StopIteration:
            # Reinitialize the iterator and stop the iteration
            self.dataloader_iter = iter(self.dataloader)
            raise StopIteration

    def reset(self):
        self.dataloader_iter = iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            raise StopIteration

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


if __name__ == "__main__":
    import os

    # Set mode for testing
    os.environ['MODE'] = 'train'
    import config

    # ------------- visualize some samples
    # Prepare test dataset
    test_dataset = TestDataset(config.image_dir, config.label_dir)  # Adjust as per your dataset class
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False)  # Adjust batch_size and other parameters as needed

    for data in test_loader:
        input = data['lr'].to(config.device)
        gt = data['gt'].to(config.device)

        print(input.shape)
        print(gt.shape)

        # # benchmark the location of the defects (dimension consistence of 2 matrix)
        # Find the indices where value is 7 along the 2nd axis
        mask = (gt == 7)
        mask = mask.sum(dim=1).squeeze()
        # Sum up 'input' along the 2nd axis to get a 2D image
        input_summed = input[:, 20:-20, :].sum(dim=1).squeeze()
        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # First subplot for 'gt_summed'
        axs[0].imshow(mask, cmap='gray')
        axs[0].set_title('Summed GT mask')
        # Second subplot for 'input_summed'
        axs[1].imshow(input_summed, cmap='gray')
        axs[1].set_title('Summed Input')

        plt.show()
