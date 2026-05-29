import copy
from torch.utils import data as data
from basicsr.data.paired_image_dataset import Dataset_PairedImage as PairedImageDataset


class ConcatLoViFDataset(data.Dataset):
    """
    Concatenate multiple LoViF track datasets into one big dataset.
    Each batch randomly samples from any of the 5 tracks.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.tracks = opt['tracks']

        self.datasets = []
        self.cum_sizes = [0]
        for name, paths in self.tracks.items():
            sub_opt = copy.deepcopy(opt)
            sub_opt['dataroot_gt'] = paths['gt']
            sub_opt['dataroot_lq'] = paths['lq']
            sub_opt['name'] = f'{opt["name"]}_{name}'
            ds = PairedImageDataset(sub_opt)
            self.datasets.append(ds)
            self.cum_sizes.append(self.cum_sizes[-1] + len(ds))

        self.total = self.cum_sizes[-1]
        print(f'[ConcatLoViFDataset] {len(self.tracks)} tracks, {self.total} total images')

    def __getitem__(self, index):
        for i in range(len(self.datasets)):
            if index < self.cum_sizes[i + 1]:
                local_idx = index - self.cum_sizes[i]
                return self.datasets[i][local_idx]
        raise IndexError

    def __len__(self):
        return self.total
