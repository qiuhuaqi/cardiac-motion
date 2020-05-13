""" Run inference on full sequence of images """
import os
import argparse
import logging
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
import nibabel as nib

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.networks import BaseNet
from model.dataset_utils import CenterCrop, Normalise, ToTensor
from model.datasets import CardiacMR_2D_Eval_UKBB, CardiacMR_2D_Inference_UKBB
from model.submodules import resample_transform
from utils.metrics import contour_distances_stack, computeJacobianDeterminant2D
from utils import xutils, flow_utils



def plot_results(target, source, warped_source, op_flow, save_path=None, title_font_size=20, show_fig=False):
    """
    Plot all motion related results in one figure,
    DVF should be normalised to [-1, 1] space
    Images should be min-max normalised to [0,1]
    """

    # convert flow into HSV flow with white background
    hsv_flow = flow_utils.flow_to_hsv(op_flow, max_mag=0.15, white_bg=True)

    ## set up the figure
    fig = plt.figure(figsize=(30, 18))
    title_pad = 10

    # source
    ax = plt.subplot(2, 4, 1)
    plt.imshow(source, cmap='gray')
    plt.axis('off')
    ax.set_title('Source', fontsize=title_font_size, pad=title_pad)

    # warped source
    ax = plt.subplot(2, 4, 2)
    plt.imshow(warped_source, cmap='gray')
    plt.axis('off')
    ax.set_title('Warped Source', fontsize=title_font_size, pad=title_pad)

    # calculate the error before and after reg
    error_before = target - source
    error_after = target - warped_source

    # error before
    ax = plt.subplot(2, 4, 3)
    plt.imshow(error_before, vmin=-2, vmax=2, cmap='seismic')
    plt.axis('off')
    ax.set_title('Error before', fontsize=title_font_size, pad=title_pad)

    # error after
    ax = plt.subplot(2, 4, 4)
    plt.imshow(error_after, vmin=-2, vmax=2, cmap='seismic')
    plt.axis('off')
    ax.set_title('Error after', fontsize=title_font_size, pad=title_pad)

    # target image
    ax = plt.subplot(2, 4, 5)
    plt.imshow(target, cmap='gray')
    plt.axis('off')
    ax.set_title('Target', fontsize=title_font_size, pad=title_pad)

    # hsv flow
    ax = plt.subplot(2, 4, 7)
    plt.imshow(hsv_flow)
    plt.axis('off')
    ax.set_title('HSV', fontsize=title_font_size, pad=title_pad)

    # quiver, or "Displacement Vector Field" (DVF)
    ax = plt.subplot(2, 4, 6)
    interval = 3  # interval between points on the grid
    background = source
    quiver_flow = np.zeros_like(op_flow)
    quiver_flow[:, :, 0] = op_flow[:, :, 0] * op_flow.shape[0] / 2
    quiver_flow[:, :, 1] = op_flow[:, :, 1] * op_flow.shape[1] / 2
    mesh_x, mesh_y = np.meshgrid(range(0, op_flow.shape[1] - 1, interval),
                                 range(0, op_flow.shape[0] - 1, interval))
    plt.imshow(background[:, :], cmap='gray')
    plt.quiver(mesh_x, mesh_y,
               quiver_flow[mesh_y, mesh_x, 1], quiver_flow[mesh_y, mesh_x, 0],
               angles='xy', scale_units='xy', scale=1, color='g')
    plt.axis('off')
    ax.set_title('DVF', fontsize=title_font_size, pad=title_pad)

    # det Jac
    ax = plt.subplot(2, 4, 8)
    jac_det, mean_grad_detJ, negative_detJ = computeJacobianDeterminant2D(op_flow)
    spec = [(0, (0.0, 0.0, 0.0)), (0.000000001, (0.0, 0.2, 0.2)),
            (0.12499999999, (0.0, 1.0, 1.0)), (0.125, (0.0, 0.0, 1.0)),
            (0.25, (1.0, 1.0, 1.0)), (0.375, (1.0, 0.0, 0.0)),
            (1, (0.94509803921568625, 0.41176470588235292, 0.07450980392156863))]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('detjac', spec)
    plt.imshow(jac_det, vmin=-1, vmax=7, cmap=cmap)
    plt.axis('off')
    ax.set_title('Jacobian (Grad: {0:.2f}, Neg: {1:.2f}%)'.format(mean_grad_detJ, negative_detJ * 100),
                 fontsize=int(title_font_size*0.9), pad=title_pad)
    # split and extend this axe for the colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(cax=cax1)
    cb.ax.tick_params(labelsize=20)

    # adjust subplot placements and spacing
    plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1, wspace=0.001, hspace=0.1)

    # saving
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=100)

    if show_fig:
        plt.show()
    plt.close()



def inference(model, subject_data_dir, eval_data, subject_output_dir, args, params):
    """
    Run inference on one subject sequence

    Args:
        model: (object) instantiated model
        subject_data_dir: (string) directory of the subject's data, absolute path
        eval_data: (dict) ED and ES images and labels to evaluate metrics
        subject_output_dir: (string) save results of the subject to this dir
        args
        params

    """
    # dataloader for one subject that loads volume pairs of two consecutive frames in a sequence
    inference_dataset = CardiacMR_2D_Inference_UKBB(subject_data_dir,
                                                    seq=params.seq,
                                                    transform=transforms.Compose([
                                                        CenterCrop(params.crop_size),
                                                        Normalise(),
                                                        ToTensor()])
                                                    )

    logging.info("Running inference computation...")

    dvf_buffer = []
    target_buffer = []
    source_buffer = []
    warped_source_buffer = []

    # loop over time frames
    for (target, source) in inference_dataset:
        # size (N, 1, H, W) to input model
        target = target.unsqueeze(1).to(device=args.device)
        source = source.unsqueeze(1).to(device=args.device)

        # run inference
        dvf = model(target, source)
        warped_source = resample_transform(source, dvf)

        # move to cpu & add to buffer, N = #slices
        dvf_buffer += [dvf.data.cpu().numpy().transpose(0, 2, 3, 1)]  # (N, H, W, 2),
        target_buffer += [target.data.squeeze(1).cpu().numpy()[:, :, :]]  # (N, H, W)
        source_buffer += [source.data.squeeze(1).cpu().numpy()[:, :, :]]  # (N, H, W)
        warped_source_buffer += [warped_source.data.squeeze(1).cpu().numpy()[:, :, :]]  # (N, H, W)

    logging.info("- Done.")

    # stack on time dimension (0) => (T, N, H, W)
    dvf_seq = np.stack(dvf_buffer, axis=0)  # (T, N, H, W, 2)
    target_seq = np.stack(target_buffer, axis=0)
    source_seq = np.stack(source_buffer, axis=0)
    warped_source_seq = np.stack(warped_source_buffer, axis=0)

    """ Save output transformation and images """
    # (optional) extract 3 slices
    num_slices = dvf_seq.shape[1]
    if not args.all_slices:
        apical_idx = int(round((num_slices - 1) * 0.75))  # 75% from basal
        mid_ven_idx = int(round((num_slices - 1) * 0.5))  # 50% from basal
        basal_idx = int(round((num_slices - 1) * 0.25))  # 25% from basal
        slices_idx = [apical_idx, mid_ven_idx, basal_idx]
    else:
        slices_idx = np.arange(0, num_slices)

    # save DVF and image sequences (original and warped)
    source_save = source_seq.transpose(2, 3, 1, 0)[..., slices_idx, :]  # (H, W, _N, T)
    warped_source_save = warped_source_seq.transpose(2, 3, 1, 0)[..., slices_idx, :]  # (H, W, _N, T)
    dvf_save = dvf_seq.transpose(2, 3, 1, 4, 0)[..., slices_idx, :, :]  # (H, W, _N, 2, T)
    dvf_save[..., 0, :] *= dvf_save.shape[0] / 2
    dvf_save[..., 1, :] *= dvf_save.shape[1] / 2  # un-normalise DVF to image pixel space

    # (note: identity image2world header matrix)
    nib.save(nib.Nifti1Image(source_save, np.eye(4)), f"{subject_output_dir}/{params.seq}.nii.gz")
    nib.save(nib.Nifti1Image(warped_source_save, np.eye(4)), f"{subject_output_dir}/warped_{params.seq}.nii.gz")
    nib.save(nib.Nifti1Image(dvf_save, np.eye(4)), f"{subject_output_dir}/{params.seq}_dvf.nii.gz")

    """"""

    """ 
    Save visual output 
    """
    if args.visual_output:
        logging.info("Saving visual outputs (WARNING: this process is slow...")

        # loop over slices
        for slice_num in slices_idx:
            logging.info("Saving results of slice no. {}".format(slice_num))
            # shape (T, H, W) or (T, H, W, 2)
            dvf_slice_seq = dvf_seq[:, slice_num, :, :]
            target_slice_seq = target_seq[:, slice_num, :, :]
            source_slice_seq = source_seq[:, slice_num, :, :]
            warped_source_slice_seq = warped_source_seq[:, slice_num, :, :]

            # set up saving directory
            output_dir_slice = os.path.join(subject_output_dir, 'slice_{}'.format(slice_num))
            if not os.path.exists(output_dir_slice):
                os.makedirs(output_dir_slice)

            # loop over time frame
            png_buffer = []
            for fr in range(dvf_slice_seq.shape[0]):
                print('Frame: {}/{}'.format(fr, dvf_slice_seq.shape[0]))
                dvf_fr = dvf_slice_seq[fr, :, :, :]
                target_fr = target_slice_seq[fr, :, :]
                source_fr = source_slice_seq[fr, :, :]
                warped_source_fr = warped_source_slice_seq[fr, :, :]

                fig_save_path = os.path.join(output_dir_slice, 'frame_{}.png'.format(fr))
                plot_results(target_fr, source_fr, warped_source_fr, dvf_fr, save_path=fig_save_path)

                # read back the PNG to save a GIF animation
                png_buffer += [imageio.imread(fig_save_path)]
            imageio.mimwrite(os.path.join(output_dir_slice, 'results.gif'), png_buffer, fps=params.fps)
    """"""


    """ 
    Evaulate motion estimation accuracy metrics for each subject
    (NOTE: only works with SAX images) 
    """
    if args.metrics:
        # unpack the ED ES data Tensor inputs, transpose from (1, N, H, W) to (N, 1, H, W)
        image_ed_batch = eval_data['image_ed_batch'].permute(1, 0, 2, 3).to(device=args.device)
        image_es_batch = eval_data['image_es_batch'].permute(1, 0, 2, 3).to(device=args.device)
        label_es_batch = eval_data['label_es_batch'].permute(1, 0, 2, 3).to(device=args.device)

        # compute optical flow and warped ed images using the trained model(source, target)
        dvf = model(image_ed_batch, image_es_batch)

        # warp ED segmentation mask to ES using nearest neighbourhood interpolation
        with torch.no_grad():
            warped_label_es_batch = resample_transform(label_es_batch.float(), dvf, interp='nearest')

        # move data to cpu to calculate metrics (also transpose into H, W, N)
        warped_label_es_batch = warped_label_es_batch.squeeze(1).cpu().numpy().transpose(1, 2, 0)
        label_es_batch = label_es_batch.squeeze(1).cpu().numpy().transpose(1, 2, 0)
        label_ed_batch = eval_data['label_ed_batch'].squeeze(0).numpy().transpose(1, 2, 0)

        # calculate contour distance metrics, metrics functions take inputs shaped in (H, W, N)
        mcd_lv, hd_lv = contour_distances_stack(warped_label_es_batch, label_ed_batch,
                                                label_class=1,
                                                dx=params.pixel_size)
        mcd_myo, hd_myo = contour_distances_stack(warped_label_es_batch, label_ed_batch,
                                                  label_class=2,
                                                  dx=params.pixel_size)
        mcd_rv, hd_rv = contour_distances_stack(warped_label_es_batch, label_ed_batch,
                                                label_class=3,
                                                dx=params.pixel_size)

        metrics = dict()
        metrics['mcd_lv'] = mcd_lv
        metrics['hd_lv'] = hd_lv
        metrics['mcd_myo'] = mcd_myo
        metrics['hd_myo'] = hd_myo
        metrics['mcd_rv'] = mcd_rv
        metrics['hd_rv'] = hd_rv

        # save the metrics to a JSON file
        metrics_save_path = os.path.join(subject_output_dir, 'metrics.json')
        xutils.save_dict_to_json(metrics, metrics_save_path)

        # save wapred ES segmentations and original (but cropped) ED segmentation into NIFTIs
        nim = nib.load(os.path.join(subject_data_dir, 'label_sa_ED.nii.gz'))
        nim_wapred_label_es = nib.Nifti1Image(warped_label_es_batch, nim.affine, nim.header)
        nib.save(nim_wapred_label_es, os.path.join(subject_output_dir, 'warped_label_ES.nii.gz'))
        nim_label_ed = nib.Nifti1Image(label_ed_batch, nim.affine, nim.header)
        nib.save(nim_label_ed, os.path.join(subject_output_dir, 'label_ED.nii.gz'))
        nim_label_es = nib.Nifti1Image(label_es_batch, nim.affine, nim.header)
        nib.save(nim_label_es, os.path.join(subject_output_dir, 'label_ES.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        default='data/inference',
                        help="Path to the dir containing inference data")

    parser.add_argument('--model_dir',
                        default=None,
                        help="Main directory for the model (with params.json)")

    parser.add_argument('--restore_file',
                        default="best.pth.tar",
                        help="Name of the file in --model_dir storing model to load before training")

    parser.add_argument('--all_slices',
                        action='store_true',
                        help="Evaluate metrics on all slices instead of only 3.")

    parser.add_argument('--no_cuda',
                        action='store_true')

    parser.add_argument('--gpu',
                        default=0,
                        help='Choose GPU')

    parser.add_argument('--num_workers',
                        default=8,
                        help='Number of dataloader workers, 0 for main process only')

    parser.add_argument('--metrics',
                        action='store_true',
                        help="Evaluating metrics")

    parser.add_argument('--save_nifti',
                        action='store_true',
                        help="Save results in NIFTI files")

    parser.add_argument('--visual_output',
                        action='store_true',
                        help="Save GIF and a sequence of PNGs of DVFs on image frames for each slice.")

    args = parser.parse_args()


    """
    Setting up
    """
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # set up logger
    xutils.set_logger(os.path.join(args.model_dir, 'inference.log'))
    logging.info(f"Running inference of model: {args.model_dir}")

    # check whether the trained model exists
    logging.info(f"Model: {args.model_dir}")
    assert os.path.exists(args.model_dir), f"No model dir found at: {args.model_dir}"

    # load setting parameters from a JSON file
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), f"No json configuration file found at: {json_path}"
    params = xutils.Params(json_path)

    # set up save dir
    output_dir = os.path.join(args.model_dir, 'inference_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    """"""


    """
    Data 
    """
    logging.info(f"Inference data path: {args.data_dir}")
    # set up the eval dataloader to evaluate metrics
    eval_dataset = CardiacMR_2D_Eval_UKBB(args.data_dir,
                                          seq=params.seq,
                                          label_prefix=params.label_prefix,
                                          transform=transforms.Compose([
                                              CenterCrop(params.crop_size),
                                              Normalise(),
                                              ToTensor()]),
                                          label_transform=transforms.Compose([
                                              CenterCrop(params.crop_size),
                                              ToTensor()])
                                          )

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=params.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=args.cuda)
    """"""


    """
    Model
    """
    # set up model and loss function
    model = BaseNet()
    model = model.to(device=args.device)

    # reload network parameters from saved model file
    logging.info(f"Loading model from saved file: "
                 f"{os.path.join(args.model_dir, args.restore_file)}")
    xutils.load_checkpoint(os.path.join(args.model_dir, args.restore_file), model)
    model.eval()
    """"""

    """
    Run inference
    """
    # loop over subjects using evaluation dataloader
    logging.info("Starting inference...")
    with tqdm(total=len(eval_dataloader)) as t:
        for idx, (image_ed_batch, image_es_batch, label_ed_batch, label_es_batch) in enumerate(eval_dataloader):
            # pack the eval data into a dict
            eval_data = dict()
            eval_data['image_ed_batch'] = image_ed_batch
            eval_data['image_es_batch'] = image_es_batch
            eval_data['label_ed_batch'] = label_ed_batch
            eval_data['label_es_batch'] = label_es_batch

            # get the subject dir from dataset
            subject_id = eval_dataloader.dataset.dir_list[idx]
            logging.info("Subject: {}".format(subject_id))

            subject_data_dir = os.path.join(args.data_dir, subject_id)
            assert os.path.exists(subject_data_dir), \
                f"Inference data of subject {subject_id} does not exist!"

            subject_output_dir = os.path.join(output_dir, subject_id)
            if not os.path.exists(subject_output_dir):
                os.makedirs(subject_output_dir)

            # run inference on the subject
            inference(model, subject_data_dir, eval_data, subject_output_dir, args, params)

            t.update()
    logging.info("Inference complete.")


