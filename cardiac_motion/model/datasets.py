import os
import os.path as path
import random
import datetime
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import torch.utils.data as data


class CardiacMR_2D_UKBB(data.Dataset):
    """
    Training class for UKBB. Loads the specific ED file as target.
    """

    def __init__(self, data_path, seq="sa", seq_length=30, augment=False, transform=None):
        # super(TrainDataset, self).__init__()
        super().__init__()  # this syntax is allowed in Python3

        self.data_path = data_path
        self.seq = seq
        self.seq_length = seq_length
        self.augment = augment
        self.transform = transform

        self.dir_list = []
        for subj_dir in sorted(os.listdir(self.data_path)):
            if path.exists(path.join(data_path, subj_dir, seq + ".nii.gz")) and path.exists(
                path.join(data_path, subj_dir, seq + "_ED.nii.gz")
            ):
                self.dir_list += [subj_dir]
            else:
                raise RuntimeError(f"Data path does not exist: {self.data_path}")

    def __getitem__(self, index):
        """
        Load and pre-process the input image.

        Args:
            index: index into the dir list

        Returns:
            target: target image, Tensor of size (1, H, W)
            source: source image sequence, Tensor of size (seq_length, H, W)
        """

        # update the seed to avoid workers sample the same augmentation parameters
        # if self.augment:
        #     np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load nifti into array
        subj_dir = os.path.join(self.data_path, self.dir_list[index])
        image_seq_path = os.path.join(subj_dir, self.seq + ".nii.gz")
        image_raw = nib.load(image_seq_path).get_data()
        image_ed_path = os.path.join(subj_dir, self.seq + "_ED.nii.gz")
        image_ed = nib.load(image_ed_path).get_data()

        if self.seq == "sa":
            # random select a z-axis slice and transpose into (seq_length, H, W)
            slice_num = random.randint(0, image_raw.shape[-2] - 1)
        else:
            slice_num = 0
        image = image_raw[:, :, slice_num, :].transpose(2, 0, 1).astype(np.float32)
        image_ed = image_ed[:, :, slice_num]

        # target images are copies of the ED frame (extended later in training code to make use of Pytorch view)
        target = image_ed[np.newaxis, :, :]  # extend dim to (1, H, W)

        # source images are a sequence of params.seq_length frames
        if image.shape[0] > self.seq_length:
            start_frame_idx = random.randint(0, image.shape[0] - self.seq_length)
            end_frame_idx = start_frame_idx + self.seq_length
            source = image[start_frame_idx:end_frame_idx, :, :]  # (seq_length, H, W)
        else:
            # if the sequence is shorter than seq_length, use the whole sequence
            source = image[1:, :, :]  # (T-1, H, W)

        # transformation functions expect input shape (N, H, W)
        if self.transform:
            target = self.transform(target)
            source = self.transform(source)

        # dev: expand target dimension (1, H, W) -> (N, H, W)
        target = target.expand(source.size()[0], -1, -1)
        return {"target": target, "source": source}

    def __len__(self):
        return len(self.dir_list)


class CardiacMR_2D_Eval_UKBB(data.Dataset):
    """Validation and evaluation for UKBB
    Fetches ED and ES frame images and segmentation labels"""

    def __init__(
        self,
        data_path,
        seq="sa",
        label_prefix="label",
        augment=False,
        transform=None,
        label_transform=None,
    ):
        super().__init__()  # this syntax is allowed in Python3

        self.data_path = data_path
        self.seq = seq
        self.label_prefix = label_prefix
        self.augment = augment

        self.transform = transform
        self.label_transform = label_transform

        # check required data files
        self.dir_list = []
        for subj_dir in sorted(os.listdir(self.data_path)):
            if (
                path.exists(path.join(data_path, subj_dir, seq + "_ES.nii.gz"))
                and path.exists(path.join(data_path, subj_dir, seq + "_ED.nii.gz"))
                and path.exists(
                    path.join(
                        data_path,
                        subj_dir,
                        "{}_".format(label_prefix) + seq + "_ED.nii.gz",
                    )
                )
                and path.exists(
                    path.join(
                        data_path,
                        subj_dir,
                        "{}_".format(label_prefix) + seq + "_ES.nii.gz",
                    )
                )
            ):
                self.dir_list += [subj_dir]

    def __getitem__(self, index):
        """
        Load and pre-process input image and label maps
        For now batch size is expected to be 1 and each batch contains
        images and labels for each subject at ED and ES (stacks)

        Args:
            index:

        Returns:
            image_ed, image_es, label_ed, label_es: Tensors of size (N, H, W)

        """
        # update the seed to avoid workers sample the same augmentation parameters
        if self.augment:
            np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load nifti into array
        image_path_ed = os.path.join(self.data_path, self.dir_list[index], self.seq + "_ED.nii.gz")
        image_path_es = os.path.join(self.data_path, self.dir_list[index], self.seq + "_ES.nii.gz")
        label_path_ed = os.path.join(
            self.data_path,
            self.dir_list[index],
            "{}_".format(self.label_prefix) + self.seq + "_ED.nii.gz",
        )
        label_path_es = os.path.join(
            self.data_path,
            self.dir_list[index],
            "{}_".format(self.label_prefix) + self.seq + "_ES.nii.gz",
        )

        # images and labels are in shape (H, W, N)
        image_ed = nib.load(image_path_ed).get_data()
        image_es = nib.load(image_path_es).get_data()
        label_ed = nib.load(label_path_ed).get_data()
        label_es = nib.load(label_path_es).get_data()

        # transpose into (N, H, W)
        image_ed = image_ed.transpose(2, 0, 1)
        image_es = image_es.transpose(2, 0, 1)
        label_ed = label_ed.transpose(2, 0, 1)
        label_es = label_es.transpose(2, 0, 1)

        # transformation functions expect input shaped (N, H, W)
        if self.transform:
            image_ed = self.transform(image_ed)
            image_es = self.transform(image_es)

        if self.label_transform:
            label_ed = self.label_transform(label_ed)
            label_es = self.label_transform(label_es)

        return image_ed, image_es, label_ed, label_es

    def __len__(self):
        return len(self.dir_list)


class CardiacMR_2D_Inference_UKBB(data.Dataset):
    """Inference dataset, works with UKBB data or data with segmentation,
    loop over frames of one subject"""

    def __init__(self, data_path, seq="sa", transform=None):
        """data_path is the path to the direcotry containing the nifti files"""
        super().__init__()  # this syntax is allowed in Python3

        self.data_path = data_path
        self.seq = seq

        self.transform = transform
        self.seq_length = None

        # load sequence image nifti
        file_path = os.path.join(self.data_path, self.seq + ".nii.gz")
        nim = nib.load(file_path)
        self.image_seq = nim.get_data()

        # pass sequence length to object handle
        self.seq_length = self.image_seq.shape[-1]

    def __getitem__(self, idx):
        """Returns volume pairs of two consecutive frames in a sequence"""

        target = self.image_seq[:, :, :, 0].transpose(2, 0, 1)
        source = self.image_seq[:, :, :, idx].transpose(2, 0, 1)

        if self.transform:
            target = self.transform(target)
            source = self.transform(source)

        return target, source

    def __len__(self):
        return self.seq_length


class CardiacMR_2D_UKBB_SynthDeform(CardiacMR_2D_UKBB):
    def __init__(self, *args, scales=(8, 16, 32), min_std=0.0, max_std=1.0, seed=None, norm_dvf=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.scales = scales
        self.min_std = min_std
        self.max_std = max_std
        self.norm_dvf = norm_dvf
        self.np_rand = np.random.default_rng(seed)  # set by worker_init_fn() passed to dataloader

    @staticmethod
    def normalise_disp(disp):
        """
        Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
        Assumes disp size is the same as the corresponding image.
        Args:
            disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field
        Returns:
            disp: (normalised disp)
        """

        ndim = disp.ndim - 2

        if type(disp) is np.ndarray:
            norm_factors = 2.0 / np.array(disp.shape[2:])
            norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

        elif type(disp) is torch.Tensor:
            norm_factors = torch.tensor(2.0) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
            norm_factors = norm_factors.view(1, ndim, *(1,) * ndim)

        else:
            raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
        return disp * norm_factors

    @classmethod
    def warp(cls, x, disp, interp_mode="bilinear", norm_disp=True):
        """
        Spatially transform an image by sampling at transformed locations (2D and 3D)

        Args:
            x: (Tensor float, shape (N, ndim, *sizes)) input image
            disp: (Tensor float, shape (N, ndim, *sizes)) dense displacement field in i-j-k order
            interp_mode: (string) mode of interpolation in grid_sample()
            norm_disp: (bool) if True, normalise the disp to [-1, 1] from voxel number space before applying

        Returns:
            deformed x, Tensor of the same shape as input
        """
        ndim = x.ndim - 2
        size = x.size()[2:]
        disp = disp.type_as(x)

        if norm_disp:
            disp = cls.normalise_disp(disp)

        # generate standard mesh grid
        grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
        grid = [grid[i].requires_grad_(False) for i in range(ndim)]

        # apply displacements to each direction (N, *size)
        warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

        # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
        warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
        warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

        return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=True)

    @classmethod
    def svf_exp(cls, flow, scale=1, steps=5, sampling="bilinear"):
        """Exponential of velocity field by Scaling and Squaring"""
        disp = flow * (scale / (2**steps))
        for i in range(steps):
            disp = disp + cls.warp(x=disp, disp=disp, interp_mode=sampling, norm_disp=True)
        return disp

    def _synthesis_deformation(
        self, out_shape, scales=16, min_std=0.0, max_std=0.5, num_batch=1, zoom_mode="bilinear"
    ) -> torch.Tensor:
        """out_shape is spatial shape, keeping batch dimension, drop if needed in datasets
        returns torch.Tensor
        """
        ndims = len(out_shape)

        # vmx generates at half resolution then upsample
        out_shape = np.asarray(out_shape, dtype=np.int32)
        gen_shape = out_shape // 2
        if np.isscalar(scales):
            scales = [scales]
        scales = [s // 2 for s in scales]

        svf = torch.zeros(num_batch, ndims, *gen_shape)

        for scale in scales:
            sample_shape = np.int32(np.ceil(gen_shape / scale))
            sample_shape = (num_batch, ndims, *sample_shape)

            std = self.np_rand.uniform(min_std, max_std, size=sample_shape)
            gauss = self.np_rand.normal(0, std, size=sample_shape)
            zoom = [o / s for o, s in zip(gen_shape, sample_shape[2:])]
            if scale > 1:
                gauss = torch.from_numpy(gauss)
                gauss = F.interpolate(gauss, scale_factor=zoom, mode=zoom_mode, align_corners=True)
            svf += gauss

        # integrate, upsample
        dvf = self.svf_exp(svf)
        dvf = F.interpolate(dvf, scale_factor=2, mode=zoom_mode, align_corners=True) * 2

        if self.norm_dvf:
            dvf = self.normalise_disp(dvf)
        return dvf

    def __getitem__(self, index):
        x_data = super().__getitem__(index)
        source = x_data["source"]
        out_shape = source.shape[1:]
        dvf = self._synthesis_deformation(
            out_shape,
            scales=self.scales,
            min_std=self.min_std,
            max_std=self.max_std,
            num_batch=source.shape[0],
        )
        target = self.warp(source.unsqueeze(1), dvf, interp_mode="bilinear", norm_disp=False).squeeze(1)
        return {"target": target, "source": source, "dvf": dvf}
