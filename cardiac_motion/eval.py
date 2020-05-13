from tqdm import tqdm
import os
import argparse
import logging
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.networks import BaseNet
from model.submodules import resample_transform
from model.dataset_utils import CenterCrop, Normalise, ToTensor
from model.datasets import CardiacMR_2D_Eval_UKBB
from utils.metrics import categorical_dice_stack, contour_distances_stack, detJac_stack
from utils import xutils



def evaluate(model, dataloader, params, args, val):
    """
    Evaluate the model on the test dataset
    Returns metrics as a dict and evaluation loss

    Args:
        model:
        dataloader:
        params:
        args:
        val: (boolean) True for validation

    Returns:

    """

    # evaluation mode
    model.eval()

    # initialise buffers
    dice_lv_buffer = []
    dice_myo_buffer = []
    dice_rv_buffer = []

    mcd_lv_buffer = []
    hd_lv_buffer = []
    mcd_myo_buffer = []
    hd_myo_buffer = []
    mcd_rv_buffer = []
    hd_rv_buffer = []

    mean_mag_grad_detJ_buffer = []
    negative_detJ_buffer = []


    with tqdm(total=len(dataloader)) as t:
        # iterate over validation subjects
        for idx, (image_ed_batch, image_es_batch, label_ed_batch, label_es_batch) in enumerate(dataloader):
            # (data all in shape of (c, N, H, W))

            # extend to (N, c, H, W)
            image_ed_batch = image_ed_batch.permute(1, 0, 2, 3).to(device=args.device)
            image_es_batch = image_es_batch.permute(1, 0, 2, 3).to(device=args.device)
            label_es_batch = label_es_batch.permute(1, 0, 2, 3).to(device=args.device)

            with torch.no_grad():
                # compute optical flow and warped ED images towards ES
                op_flow = model(image_ed_batch, image_es_batch)

                # transform label mask of ES frame
                warped_label_es_batch = resample_transform(label_es_batch.float(), op_flow, interp='nearest')


            """ Move data to device """
            if args.cuda:
                # move data to cpu to calculate metrics
                # (the axis permutation is to comply with metric calculation code which takes input shape H, W, N)
                warped_label_es_batch = warped_label_es_batch.squeeze(1).cpu().numpy().transpose(1, 2, 0)
                label_ed_batch = label_ed_batch.squeeze(0).numpy().transpose(1, 2, 0)
                op_flow = op_flow.data.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
            else:
                # CPU version of the code
                warped_label_es_batch = warped_label_es_batch.squeeze(1).numpy().transpose(1, 2, 0)
                label_ed_batch = label_ed_batch.squeeze(0).numpy().transpose(1, 2, 0)
                op_flow = op_flow.data.numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
            """"""

            """ Calculate the metrics (only works with SAX images) """
            # (optional) extract 3 slices (apical, mid-ventricle and basal)
            if not args.all_slices:
                num_slices = label_ed_batch.shape[-1]
                apical_idx = int(round((num_slices - 1) * 0.75))  # 75% from basal
                mid_ven_idx = int(round((num_slices - 1) * 0.5))  # 50% from basal
                basal_idx = int(round((num_slices - 1) * 0.25))  # 25% from basal
                slices_idx = [apical_idx, mid_ven_idx, basal_idx]

                warped_label_es_batch = warped_label_es_batch[:, :, slices_idx]
                label_ed_batch = label_ed_batch[:, :, slices_idx]
                op_flow = op_flow[slices_idx, :, :, :]  # needed for detJac

            # dice
            dice_lv = categorical_dice_stack(warped_label_es_batch, label_ed_batch, label_class=1)
            dice_myo = categorical_dice_stack(warped_label_es_batch, label_ed_batch, label_class=2)
            dice_rv = categorical_dice_stack(warped_label_es_batch, label_ed_batch, label_class=3)

            dice_lv_buffer += [dice_lv]
            dice_myo_buffer += [dice_myo]
            dice_rv_buffer += [dice_rv]

            # contour distances
            mcd_lv, hd_lv = contour_distances_stack(warped_label_es_batch, label_ed_batch, label_class=1, dx=params.pixel_size)
            mcd_myo, hd_myo = contour_distances_stack(warped_label_es_batch, label_ed_batch, label_class=2, dx=params.pixel_size)
            mcd_rv, hd_rv = contour_distances_stack(warped_label_es_batch, label_ed_batch, label_class=3, dx=params.pixel_size)

            # determinant of Jacobian
            mean_grad_detJ, mean_negative_detJ = detJac_stack(op_flow)


            # update buffers
            mcd_lv_buffer += [mcd_lv]
            hd_lv_buffer += [hd_lv]
            mcd_myo_buffer += [mcd_myo]
            hd_myo_buffer += [hd_myo]
            mcd_rv_buffer += [mcd_rv]
            hd_rv_buffer += [hd_rv]

            mean_mag_grad_detJ_buffer += [mean_grad_detJ]
            negative_detJ_buffer += [mean_negative_detJ]

            t.update()

    # construct metrics dict
    metrics = {'dice_lv_mean': np.mean(dice_lv_buffer), 'dice_lv_std': np.std(dice_lv_buffer),
               'dice_myo_mean': np.mean(dice_myo_buffer), 'dice_myo_std': np.std(dice_myo_buffer),
               'dice_rv_mean': np.mean(dice_rv_buffer), 'dice_rv_std': np.std(dice_rv_buffer),

               'mcd_lv_mean': np.mean(mcd_lv_buffer), 'mcd_lv_std': np.std(mcd_lv_buffer),
               'mcd_myo_mean': np.mean(mcd_myo_buffer), 'mcd_myo_std': np.std(mcd_myo_buffer),
               'mcd_rv_mean': np.mean(mcd_rv_buffer), 'mcd_rv_std': np.std(mcd_rv_buffer),

               'hd_lv_mean': np.mean(hd_lv_buffer), 'hd_lv_std': np.std(hd_lv_buffer),
               'hd_myo_mean': np.mean(hd_myo_buffer), 'hd_myo_std': np.std(hd_myo_buffer),
               'hd_rv_mean': np.mean(hd_rv_buffer), 'hd_rv_std': np.std(hd_rv_buffer),

               'mean_mag_grad_detJ_mean': np.mean(mean_mag_grad_detJ_buffer),
               'mean_mag_grad_detJ_std': np.std(mean_mag_grad_detJ_buffer),

               'negative_detJ_mean': np.mean(negative_detJ_buffer),
               'negative_detJ_std': np.std(negative_detJ_buffer)
               }


    if not val:
        # testing only: save all metrics evaluated for all test subjects in pandas dataframe
        test_result_dir = os.path.join(args.model_dir, "test_results")
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)

        # save metrics results mean & std
        xutils.save_dict_to_json(metrics,
                                 f"{test_result_dir}/test_results_3slices_{not args.all_slices}.json")

        # save accuracy metrics of every subject
        subj_id_buffer = dataloader.dataset.dir_list
        df_buffer = []
        column_method = ['DL'] * len(subj_id_buffer)
        for struct in ['LV', 'MYO', 'RV']:
            if struct == 'LV':
                ls_dice = dice_lv_buffer
                ls_mcd = mcd_lv_buffer
                ls_hd = hd_lv_buffer
            elif struct == 'MYO':
                ls_dice = dice_myo_buffer
                ls_mcd = mcd_myo_buffer
                ls_hd = hd_myo_buffer
            elif struct == 'RV':
                ls_dice = dice_rv_buffer
                ls_mcd = mcd_rv_buffer
                ls_hd = hd_rv_buffer

            ls_struct = [struct] * len(subj_id_buffer)
            data = {'Method': column_method,
                    'ID': subj_id_buffer,
                    'Structure': ls_struct,
                    'Dice': ls_dice,
                    'MCD': ls_mcd,
                    'HD': ls_hd}
            df_buffer += [pd.DataFrame(data=data)]
        # concatenate df and save
        metrics_df = pd.concat(df_buffer, axis=0)
        metrics_df.to_pickle(f"{test_result_dir}/test_accuracy_results_3slices_{not args.all_slices}.pkl")

        # save detJac metrics for every subject
        jac_data = {'Method': column_method,
                    'ID': subj_id_buffer,
                    'GradDetJac': mean_mag_grad_detJ_buffer,
                    'NegDetJac': negative_detJ_buffer}
        jac_df = pd.DataFrame(data=jac_data)
        jac_df.to_pickle(f"{test_result_dir}/test_Jacobian_results_3slices{not args.all_slices}.pkl")

    return metrics



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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
    xutils.set_logger(os.path.join(args.model_dir, 'eval.log'))
    logging.info(f"Running evaluation of model: \n\t{args.model_dir}")

    # check whether the trained model exists
    assert os.path.exists(args.model_dir), f"No model dir found at {args.model_dir}"


    # load setting parameters from a JSON file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = xutils.Params(json_path)
    """"""


    """
    Data
    """
    # set dataset and DataLoader
    logging.info(f"Eval data path: \n\t{params.eval_data_path}")
    eval_dataset = CardiacMR_2D_Eval_UKBB(params.eval_data_path,
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
    logging.info(f"Loading model from saved file: \n\t{os.path.join(args.model_dir, args.restore_file)}")
    xutils.load_checkpoint(os.path.join(args.model_dir, args.restore_file), model)
    """"""


    """ 
    Run the evaluation and calculate the metrics 
    """
    logging.info("Running evaluation...")
    evaluate(model, eval_dataloader, params, args, val=False)
    logging.info(f"Evaluation complete. Model: {args.model_dir}")
    """"""
