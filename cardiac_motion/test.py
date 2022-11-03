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

STRUCTURES = ["lv", "myo", "rv"]
SEG_METRICS = ["dice", "mcd", "hd"]
DVF_METRICS = ["mean_mag_grad_detJ", "negative_detJ"]
METRICS = [
    f"{metric}_{struct}" for metric in SEG_METRICS for struct in STRUCTURES
] + DVF_METRICS


def test(
    model,
    dataloader,
    pixel_size=1.0,
    all_slices=False,
    run_inference=True,
    run_eval=True,
    device=torch.device("cpu"),
    save_metric_results=False,
):
    """Run model inference on test dataset"""
    model.eval()

    # initialise metric result dictionary
    metric_results_lists = {metric: [] for metric in METRICS}

    with tqdm(total=len(dataloader)) as t:
        for idx, (
            image_ed_batch,
            image_es_batch,
            label_ed_batch,
            label_es_batch,
        ) in enumerate(dataloader):
            # (c, N, H, W) to (N, c, H, W)
            image_ed_batch = image_ed_batch.permute(1, 0, 2, 3).to(device=device)
            image_es_batch = image_es_batch.permute(1, 0, 2, 3).to(device=device)
            label_es_batch = label_es_batch.permute(1, 0, 2, 3).to(device=device)

            with torch.no_grad():
                # compute dvf and warped ED images towards ES
                dvf = model(image_ed_batch, image_es_batch)

                # transform label mask of ES frame
                warped_label_es_batch = resample_transform(
                    label_es_batch.float(), dvf, interp="nearest"
                )

            # Move data to cpu and numpy
            warped_label_es_batch = (
                warped_label_es_batch.squeeze(1).cpu().numpy().transpose(1, 2, 0)
            )  # (H, W, N)
            label_ed_batch = (
                label_ed_batch.squeeze(0).numpy().transpose(1, 2, 0)
            )  # (H, W, N)
            dvf = dvf.data.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)

            if not all_slices:
                # extract 3 slices (apical, mid-ventricle and basal)
                num_slices = label_ed_batch.shape[-1]
                apical_idx = int(round((num_slices - 1) * 0.75))  # 75% from basal
                mid_ven_idx = int(round((num_slices - 1) * 0.5))  # 50% from basal
                basal_idx = int(round((num_slices - 1) * 0.25))  # 25% from basal
                slices_idx = [apical_idx, mid_ven_idx, basal_idx]

                warped_label_es_batch = warped_label_es_batch[:, :, slices_idx]
                label_ed_batch = label_ed_batch[:, :, slices_idx]
                dvf = dvf[slices_idx, :, :, :]  # needed for detJac

            # accumulate metric results
            metrics_result_per_batch = evaluate_per_batch(
                warped_label_es_batch, label_ed_batch, dvf, pixel_size=pixel_size
            )
            for k in metric_results_lists.keys():
                metric_results_lists[k].append(metrics_result_per_batch[k])

            t.update()

            ## debug
            if idx == 5:
                break

    # reduce metrics results to mean and std
    metric_results_mean_std = {}
    for metric, result_list in metric_results_lists.items():
        metric_results_mean_std[f"{k}_mean"] = np.mean(result_list)
        metric_results_mean_std[f"{k}_std"] = np.std(result_list)

    if save_metric_results:
        # save all metrics evaluated for all test subjects in pandas dataframe
        test_result_dir = os.path.join(args.model_dir, "test_results")
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)

        logging.info(f"Saving metric results at: {test_result_dir}")

        # save metrics results mean & std
        xutils.save_dict_to_json(
            metric_results_mean_std,
            f"{test_result_dir}/test_results_3slices_{not all_slices}.json",
        )

        # save accuracy metrics of every subject
        subj_id_buffer = dataloader.dataset.dir_list
        df_buffer = []
        column_method = ["DL"] * len(subj_id_buffer)
        for struct in STRUCTURES:
            ls_struct = [struct] * len(subj_id_buffer)
            seg_metric_data = {
                "Method": column_method,
                "ID": subj_id_buffer,
                "Structure": ls_struct,
            }
            for metric in SEG_METRICS:
                seg_metric_data[metric] = metric_results_lists[metric]
            df_buffer += [pd.DataFrame(data=seg_metric_data)]

        # concatenate df and save
        metrics_df = pd.concat(df_buffer, axis=0)
        metrics_df.to_pickle(
            f"{test_result_dir}/test_accuracy_results_3slices_{not all_slices}.pkl"
        )

        # save detJac metrics for every subject
        jac_metric_data = {
            "Method": column_method,
            "ID": subj_id_buffer,
        }
        for metric in DVF_METRICS:
            jac_metric_data[metric] = metric_results_lists[metric]
        jac_df = pd.DataFrame(data=jac_metric_data)
        jac_df.to_pickle(
            f"{test_result_dir}/test_Jacobian_results_3slices{not all_slices}.pkl"
        )

    return metric_results_mean_std


def evaluate_per_batch(warped_label_es_batch, label_ed_batch, dvf, pixel_size=1.0):
    metric_results = {metric: 0.0 for metric in METRICS}
    # dice
    for cls, struct in enumerate(STRUCTURES):
        metric_results[f"dice_{struct}"] = categorical_dice_stack(
            warped_label_es_batch, label_ed_batch, label_class=cls + 1
        )

    # contour distances
    for cls, struct in enumerate(STRUCTURES):
        (
            metric_results[f"mcd_{struct}"],
            metric_results[f"hd_{struct}"],
        ) = contour_distances_stack(
            warped_label_es_batch,
            label_ed_batch,
            label_class=cls + 1,
            dx=pixel_size,
        )

    # dvf regularity and smoothness metrics
    (
        metric_results["mean_grad_detJ"],
        metric_results["mean_negative_detJ"],
    ) = detJac_stack(dvf)
    return metric_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Main directory for the model (with params.json)",
    )

    parser.add_argument(
        "--restore_file",
        default="best.pth.tar",
        help="Name of the file in --model_dir storing model to load before training",
    )

    parser.add_argument(
        "--all_slices",
        action="store_true",
        help="Evaluate metrics on all slices instead of 3 (75%/50%/30%) tran-axial slices.",
    )

    parser.add_argument("--no_inference", action="store_true")

    parser.add_argument("--no_eval", action="store_true")

    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument("--gpu", default=0, help="Choose GPU")

    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of dataloader workers, 0 for main process only",
    )

    args = parser.parse_args()

    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # set up logger
    xutils.set_logger(os.path.join(args.model_dir, "eval.log"))
    logging.info(f"Running evaluation of model: \n\t{args.model_dir}")

    # check whether the trained model exists
    assert os.path.exists(args.model_dir), f"No model dir found at {args.model_dir}"

    # load setting parameters from a JSON file
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = xutils.Params(json_path)

    # set dataset and DataLoader
    logging.info(f"Eval data path: \n\t{params.eval_data_path}")
    eval_dataset = CardiacMR_2D_Eval_UKBB(
        params.eval_data_path,
        seq=params.seq,
        label_prefix=params.label_prefix,
        transform=transforms.Compose(
            [CenterCrop(params.crop_size), Normalise(), ToTensor()]
        ),
        label_transform=transforms.Compose([CenterCrop(params.crop_size), ToTensor()]),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
    )

    # set up model and loss function
    model = BaseNet()
    model = model.to(device=args.device)

    # load network parameters from saved checkpoint
    logging.info(
        f"Loading model from saved file: \n\t{os.path.join(args.model_dir, args.restore_file)}"
    )
    xutils.load_checkpoint(os.path.join(args.model_dir, args.restore_file), model)

    logging.info("Running evaluation...")
    test(
        model,
        eval_dataloader,
        pixel_size=params.pixel_size,
        all_slices=args.all_slices,
        device=args.device,
        save_metric_results=False,
    )
    logging.info(f"Evaluation complete. Model: {args.model_dir}")
