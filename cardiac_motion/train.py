from tqdm import tqdm
import os
import shutil
import argparse
import logging

import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.networks import BaseNet
from model.losses import huber_loss_spatial
from model.dataset_utils import CenterCrop, Normalise, ToTensor
from model.datasets import CardiacMR_2D_UKBB, CardiacMR_2D_Eval_UKBB, CardiacMR_2D_UKBB_SynthDeform
from model.submodules import resample_transform

from test import test
from utils import xutils, flow_utils
from utils.visualise import visualise_result

torch.manual_seed(7)


def train_epoch(model, optimizer, dataloader, args, params, epoch, summary_writer):
    """
    Train the model for one epoch.V

    Args:
        model: (torch.nn.Module instance) the neural network
        optimizer: (torch.optim instance) optimizer for parameters of model
        dataloader: (DataLoader instance) a torch.utils.data.DataLoader object that fetches training data
        params: (Params instance) configuration parameters
        epoch: (int) number of epoch this is training (for the summary writer)
        summary_writer: TensorBoardX SummaryWriter()
    """

    # training mode
    model.train()

    with tqdm(total=len(dataloader)) as t:
        for it, x_data in enumerate(dataloader):
            # (N=seq_length, 1/2, H, W)
            target = x_data["target"].permute(1, 0, 2, 3).to(device=args.device)
            source = x_data["source"].permute(1, 0, 2, 3).to(device=args.device)

            # forward pass and compute loss
            dvf = model(target, source)

            # loss
            mse_fn = torch.nn.MSELoss()
            if args.supervised_synth:
                dvf_gt = x_data["dvf"].squeeze(0).to(device=args.device)
                loss = mse_fn(dvf, dvf_gt)
                losses = {"mse_dvf": loss}
            else:
                warped_source = resample_transform(source, dvf)
                sim_loss = mse_fn(target, warped_source)
                reg_loss = huber_loss_spatial(dvf) * params.reg_weight
                loss = sim_loss + reg_loss
                losses = {"mse": sim_loss, "huber_spatial": reg_loss}

            # backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save summary of loss every some steps
            if it % params.save_summary_steps == 0:
                summary_writer.add_scalar("loss", loss.data, global_step=epoch * len(dataloader) + it)

                for loss_name, loss_value in losses.items():
                    summary_writer.add_scalar(
                        "losses/{}".format(loss_name),
                        loss_value.data,
                        global_step=epoch * len(dataloader) + it,
                    )

            # update tqdm, show the loss value after the progress bar
            t.set_postfix(loss="{:05.3f}".format(loss.data))
            t.update()

            # save visualisation of training results
            if (epoch + 1) % params.save_result_epochs == 0 or (epoch + 1) == params.num_epochs:
                if it == len(dataloader) - 1:  # at the end of the epoch
                    # warp source image with full resolution dvf
                    warped_source = resample_transform(source, dvf)

                    dvf_np = dvf.data.cpu().numpy().transpose(0, 2, 3, 1)  # (N, H, W, 2)
                    dvf_np *= target.shape[-1] / 2
                    warped_source = warped_source.data.cpu().numpy()[:, 0, :, :]  # (N, H, W)

                    target = target.data.cpu().numpy()[:, 0, :, :]  # (N, H, W)
                    source = source.data.cpu().numpy()[:, 0, :, :]  # (N, H, W), here N = frames -1

                    # # set up the result dir for this epoch
                    # save_result_dir = os.path.join(args.model_dir, "train_results", "epoch_{}".format(epoch + 1))
                    # if not os.path.exists(save_result_dir):
                    #     os.makedirs(save_result_dir)
                    #
                    # # NOTE: the following code saves all N frames in a batch
                    # # save dvf (hsv + quiver), target, source, warped source and error
                    # # flow_utils.save_flow_hsv(op_flow, target, save_result_dir, fps=params.fps)
                    # flow_utils.save_warp_n_error(warped_source, target, source, save_result_dir, fps=params.fps)
                    # flow_utils.save_flow_quiver(
                    #     dvf_np,
                    #     source,
                    #     save_result_dir,
                    #     fps=params.fps,
                    # )

                    # visualise in Tensorboard
                    vis_data_dict = {
                        "target": target[:, np.newaxis, ...],
                        "source": source[:, np.newaxis, ...],
                        "target_pred": warped_source[:, np.newaxis, ...],
                        "warped_source": warped_source[:, np.newaxis, ...],
                        "disp_pred": dvf_np.transpose(0, 3, 1, 2),
                    }

                    if "dvf" in x_data.keys():
                        vis_data_dict["disp_gt"] = x_data["dvf"].squeeze(0).numpy() * target.shape[-1] / 2

                    train_fig = visualise_result(data_dict=vis_data_dict)
                    summary_writer.add_figure(
                        "training_fig", train_fig, global_step=epoch * len(dataloader) + it, close=True
                    )


def train(model, optimizer, dataloaders, args, params):
    """
    Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        dataloaders: (dict) train and val dataloaders
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that transforms image with dvf prediction and compute loss
        params: (instance of Params) configuration parameters
    """

    # reload weights from a specified file to resume training
    if args.restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file)
        logging.info("Restoring parameters from {}".format(restore_path))
        xutils.load_checkpoint(restore_path, model, optimizer)

    # set up TensorboardX summary writers
    train_summary_writer = xutils.set_summary_writer(args.model_dir, "train")
    val_summary_writer = xutils.set_summary_writer(args.model_dir, "val")

    # unpack dataloaders
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]

    # Training loop
    for epoch in range(params.num_epochs):
        logging.info("Epoch number {}/{}".format(epoch + 1, params.num_epochs))

        # train the model for one epoch
        logging.info("Training...")
        train_epoch(model, optimizer, train_dataloader, args, params, epoch, train_summary_writer)

        # validation
        if (epoch + 1) % params.val_epochs == 0 or (epoch + 1) == params.num_epochs:
            logging.info("Validating at epoch: {} ...".format(epoch + 1))
            val_metrics = test(
                model,
                val_dataloader,
                args.model_dir,
                pixel_size=params.pixel_size,
                save_output=False,
                run_eval=True,
                save_metric_results=False,
                log_visual_tb=True,
                summary_writer=val_summary_writer,
                device=args.device,
            )

            # save the most recent results in a JSON file
            save_path = os.path.join(args.model_dir, f"val_results_last_3slices_{not args.all_slices}.json")
            xutils.save_dict_to_json(val_metrics, save_path)

            logging.info("Mean val dice: {:05.3f}".format(val_metrics["dice_mean"]))
            logging.info("Mean val mcd: {:05.3f}".format(val_metrics["mcd_mean"]))
            logging.info("Mean val hd: {:05.3f}".format(val_metrics["hd_mean"]))
            logging.info("Mean val negative detJ: {:05.3f}".format(val_metrics["negative_detJ_mean"]))
            logging.info("Mean val mag grad detJ: {:05.3f}".format(val_metrics["mean_mag_grad_detJ_mean"]))
            assert val_metrics["negative_detJ_mean"] <= 1, "Invalid det Jac: Ratio of folding points > 1"

            # determine if the best model
            is_best = False
            current_one_metric = val_metrics["dice_mean"]  # use mean val dice to choose best model
            if epoch + 1 == params.val_epochs:  # first validation
                best_one_metric = current_one_metric
            if current_one_metric >= best_one_metric:
                is_best = True
                best_one_metric = current_one_metric

            # save model checkpoint
            xutils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                },
                is_best=is_best,
                checkpoint=args.model_dir,
            )

            for key, value in val_metrics.items():
                val_summary_writer.add_scalar(
                    "metrics/{}".format(key),
                    value,
                    global_step=epoch * len(train_dataloader),
                )

            # save the validation results for the best model separately
            if is_best:
                save_path = os.path.join(
                    args.model_dir,
                    f"val_results_best_3slices_{not args.all_slices}.json",
                )
                xutils.save_dict_to_json(val_metrics, save_path)

    # close TensorBoard summary writers
    train_summary_writer.close()
    val_summary_writer.close()


def seed_numpy_rng(worker_id):
    """Pytorch dataloader worker_init_fn to seed numpy rng differently for each worker"""
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed
    dataset = worker_info.dataset
    dataset.np_rand = np.random.default_rng(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Main directory for the model (with params.json)",
    )

    parser.add_argument(
        "--supervised_synth",
        action="store_true",
        help="Supervised training with synthetic deformaiton ground truth if given",
    )

    parser.add_argument("--synth_max_std", default=2.0, type=float, help="Maximal std when synthesising deformation")

    parser.add_argument(
        "--restore_file",
        default=None,
        help="(Optional) Name of the file in --model_dir storing model to load before training",
    )

    parser.add_argument(
        "--all_slices",
        action="store_true",
        help="Evaluate metrics on all slices instead of only 3.",
    )

    parser.add_argument("--no_cuda", action="store_true")

    parser.add_argument("--gpu", default=0, help="Choose GPU")

    parser.add_argument("--seed", default=7, help="RNG seed")

    parser.add_argument(
        "--num_workers",
        default=0,
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

    # set up model dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # set up the logger
    xutils.set_logger(os.path.join(args.model_dir, "train.log"))
    logging.info("Model: {}".format(args.model_dir))

    # load setting parameters from a JSON file
    json_path = os.path.join(args.model_dir, "params.json")
    if not os.path.exists(json_path):
        logging.info(f"No JSON configuration file found at {json_path}, initialising with default params.json...")
        default_json_path = f"{os.getcwd()}/params.json"
        shutil.copy(default_json_path, json_path)
    params = xutils.Params(json_path)

    # set up dataset and DataLoader
    logging.info("Setting up data loaders...")
    dataloaders = {}

    if args.supervised_synth:
        train_dataset = CardiacMR_2D_UKBB_SynthDeform(
            params.train_data_path,
            max_std=args.synth_max_std,
            seq=params.seq,
            seq_length=params.seq_length,
            transform=transforms.Compose([CenterCrop(params.crop_size), Normalise(), ToTensor()]),
            seed=args.seed,
        )
    else:
        train_dataset = CardiacMR_2D_UKBB(
            params.train_data_path,
            seq=params.seq,
            seq_length=params.seq_length,
            transform=transforms.Compose([CenterCrop(params.crop_size), Normalise(), ToTensor()]),
        )
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
        worker_init_fn=seed_numpy_rng,
    )

    val_dataset = CardiacMR_2D_Eval_UKBB(
        params.val_data_path,
        seq=params.seq,
        label_prefix=params.label_prefix,
        transform=transforms.Compose([CenterCrop(params.crop_size), Normalise(), ToTensor()]),
        label_transform=transforms.Compose([CenterCrop(params.crop_size), ToTensor()]),
    )
    dataloaders["val"] = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
    )
    logging.info("- Done.")

    # model and optimiser
    model = BaseNet()
    model = model.to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    # run
    logging.info("Starting training and validation for {} epochs.".format(params.num_epochs))
    train(model, optimizer, dataloaders, args, params)
    logging.info("Training and validation complete.")
