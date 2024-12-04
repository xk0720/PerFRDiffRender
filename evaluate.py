import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import logging
from utils.render import Render
from metric import *
from dataset.dataset import get_dataloader
from utils.util import load_config, init_seed, get_logging_path
import model as module_arch


def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    # for loading the trained-model weights.
    parser.add_argument("--epoch_num", type=int, help="epoch number of saving model weight", required=True)
    parser.add_argument("--exp_num", type=int, help="the number of training experiment.", required=True)
    parser.add_argument("--mode", type=str, help="train (val) or test", required=True)
    parser.add_argument("--config", type=str, help="config path", required=True)
    parser.add_argument("--evaluate_log_dir", type=str, default="./log/evaluate")  # evaluate
    parser.add_argument('--test_period', type=int, default=10)
    args = parser.parse_args()
    return args


def compute_mse(prediction, target):
    _, k, seq_len, dim = prediction.shape
    # join last two dimensions of prediction and target
    prediction = prediction.reshape(-1, k, seq_len * dim)
    target = target.reshape(-1, k, seq_len * dim)
    loss = ((prediction - target) ** 2).mean(axis=-1)  # (batch_size, k)
    return torch.mean(loss)


def evaluate(cfg, device, model, test_loader, render, split):
    model.eval()

    iteration = 0
    out_dir = os.path.join(cfg.trainer.out_dir, split, 'exp_' + str(cfg.exp_num))
    os.makedirs(out_dir, exist_ok=True)

    speaker_3dmm_list = []
    listener_3dmm_gt_list = []
    listener_3dmm_pred_list = []

    for batch_idx, (
            speaker_audio_clip,
            speaker_video_clip,
            speaker_emotion_clip,
            speaker_3dmm_clip,
            listener_video_clip,
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            listener_reference,
    ) in enumerate(tqdm(test_loader)):
        _3dmm_dim = listener_3dmm_clip.shape[-1]

        # if batch_idx not in [110, 120, 460, 490, 580]:
        #     continue

        (speaker_audio_clip,  # (bs, token_len, 78)
         speaker_video_clip,  # (bs, token_len, 3, 224, 224)
         speaker_emotion_clip,  # (bs, token_len, 25)
         speaker_3dmm_clip,  # (bs, token_len, 58)
         listener_video_clip,  # (bs, token_len, 3, 224, 224)
         listener_emotion_clip,  # (bs, token_len, 25)
         listener_3dmm_clip,  # (bs, token_len, 58)
         listener_3dmm_clip_personal,  # (bs * k, token_len, 58)
         listener_reference) = \
            (speaker_audio_clip.to(device),
             speaker_video_clip.to(device),
             speaker_emotion_clip.to(device),
             speaker_3dmm_clip.to(device),
             listener_video_clip.to(device),
             listener_emotion_clip.to(device),
             listener_3dmm_clip.to(device),
             listener_3dmm_clip_personal.to(device),
             listener_reference.to(device))  # (bs, 3, 224, 224)

        # listener_3dmm_clip_save = listener_3dmm_clip.detach().clone().cpu()
        # just for dimension compatibility during inference
        listener_emotion_clip = listener_emotion_clip.repeat_interleave(
            cfg.test_dataset.k_appro, dim=0)  # (bs * k, token_len, 25)
        listener_3dmm_clip = listener_3dmm_clip.repeat_interleave(
            cfg.test_dataset.k_appro, dim=0)  # (bs * k, token_len, 58)

        if (batch_idx % cfg.test_period) == 0:
            with torch.no_grad():

                # import time
                # start_time = time.time()
                # print('starting feed-forward step.')

                _, listener_3dmm_pred = model(
                    speaker_audio=speaker_audio_clip,
                    speaker_emotion_input=speaker_emotion_clip,
                    speaker_3dmm_input=speaker_3dmm_clip,
                    listener_emotion_input=listener_emotion_clip,
                    listener_3dmm_input=listener_3dmm_clip,
                    listener_personal_input=listener_3dmm_clip_personal,
                )
                # listener_3dmm_pred["prediction_3dmm"].shape: (bs, k_appro==10, seq_len==750, 3dmm_dim==58)
                listener_3dmm_pred = listener_3dmm_pred["prediction_3dmm"]

                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"Time for one feed-forward step: {elapsed_time:.6f} seconds")

                # TODO: Rendering!
                render.rendering_for_fid(
                    out_dir,
                    "{}_iter_{}".format(split, str(batch_idx + 1)),
                    listener_3dmm_pred[0, 0],  # (750, 58)
                    speaker_video_clip[0],  # (750, 3, 224, 224)
                    listener_reference[0],  # (3, 224, 224)
                    listener_video_clip[0],  # (750, 3, 224, 224)
                    step=1,  # set frame step to 10.
                )

                speaker_3dmm_list.append(speaker_3dmm_clip.detach().cpu())
                listener_3dmm_pred_list.append(listener_3dmm_pred.detach().cpu())
                listener_3dmm_gt_list.append(listener_3dmm_clip.detach().cpu())

    # all_speaker_3dmm = torch.cat(speaker_3dmm_list, dim=0)
    # shape: (N, 750, 3dmm_dim)
    all_listener_3dmm_pred = torch.cat(listener_3dmm_pred_list, dim=0)
    # shape: (N, 10, 750, 3dmm_dim)
    all_listener_3dmm_gt = torch.cat(listener_3dmm_gt_list, dim=0)
    # shape: (N * 10, 750, 3dmm_dim)

    # TODO: debug: compute MSE for inference data.
    MSE = compute_mse(all_listener_3dmm_pred, all_listener_3dmm_gt)
    print("MSE over all inference data is {}".format(MSE.item()))
    # logging.info("MSE over all inference data is {}".format(MSE.item()))


def main(args):
    # load yaml config
    cfg = load_config(args=args, config_path=args.config)
    init_seed(seed=cfg.trainer.seed)  # seed initialization

    # logging
    logging_path = get_logging_path(cfg.evaluate_log_dir)
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    test_loader = get_dataloader(cfg.test_dataset)
    split = cfg.test_dataset.split

    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')  # Adjust the device ordinal as needed
        render = Render('cuda')
    else:
        device = torch.device('cpu')
        render = Render()

    model = getattr(module_arch, cfg.trainer.model)(cfg, device)
    model.to(device)

    logging.info("-----------------Start Rendering-----------------")

    evaluate(cfg, device, model, test_loader, render, split)  # render

    logging.info("-----------------Finish Rendering-----------------")


if __name__ == '__main__':
    main(args=parse_arg())
