import os
import logging
import torch
import argparse
from torch import optim
from functools import partial
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from dataset.dataset import get_dataloader
import model.diffusion.utils.losses as module_loss
from torch.utils.tensorboard import SummaryWriter
from utils.util import load_config, init_seed, get_logging_path, get_tensorboard_path, AverageMeter, collect_grad_stats
import model as module_arch
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--exp_num", type=int, help="the number of the experiment.", required=True)
    parser.add_argument("--mode", type=str, help="train (val) or test", required=True)
    parser.add_argument("--writer", type=bool, help="whether use tensorboard", required=True)
    parser.add_argument("--config", type=str, help="config path", required=True)
    args = parser.parse_args()
    return args


def save_checkpoint(cfg, net, optimizer, epoch, is_best=False):  # hypernet
    checkpoint = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_dir = os.path.join(cfg.trainer.saving_checkpoint_dir, net.get_model_name(), 'exp_' + str(cfg.exp_num))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'epoch_' + "{:03d}".format(epoch) + '_checkpoint.pth')
    torch.save(checkpoint, save_path)

    if is_best:
        save_path = os.path.join(save_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, save_path)


def optimizer_resume(cfg, net, optimizer):
    # load model weights and optimizer
    model_name = net.get_model_name()
    save_path = os.path.join(cfg.trainer.saving_checkpoint_dir, model_name, cfg.main_model.args.resume)
    checkpoint = torch.load(save_path, map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Successfully resume optimizer!")


def train(cfg, model, train_loader, optimizer_hypernet, optimizer_mainnet, criterion, epoch, writer, device):
    whole_losses = AverageMeter()
    diff_prior_losses = AverageMeter()
    diff_decoder_losses = AverageMeter()
    # regular_losses = AverageMeter()

    model.train()
    for batch_idx, (
            speaker_audio_clip,
            speaker_video_clip,  # (bs, token_len, 3, 224, 224)
            speaker_emotion_clip,
            speaker_3dmm_clip,
            listener_video_clip,  # (bs, token_len, 3, 224, 224)
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            listener_reference,
    ) in enumerate(tqdm(train_loader)):

        batch_size = speaker_audio_clip.shape[0]
        (speaker_audio_clip,  # (bs, token_len, 78)
         speaker_emotion_clip,  # (bs, token_len, 25)
         speaker_3dmm_clip,  # (bs, token_len, 58)
         listener_emotion_clip,  # (bs * k, token_len, 25)
         listener_3dmm_clip,  # (bs * k, token_len, 58)
         listener_3dmm_clip_personal,  # (bs * k, token_len, 58)
         listener_reference) = \
            (speaker_audio_clip.to(device),
             speaker_emotion_clip.to(device),
             speaker_3dmm_clip.to(device),
             listener_emotion_clip.to(device),
             listener_3dmm_clip.to(device),
             listener_3dmm_clip_personal.to(device),
             listener_reference.to(device))  # (bs, 3, 224, 224)

        input_dict = {
            "speaker_audio": speaker_audio_clip,
            "speaker_emotion_input": speaker_emotion_clip,
            "speaker_3dmm_input": speaker_3dmm_clip,
            "listener_emotion_input": listener_emotion_clip,
            "listener_3dmm_input": listener_3dmm_clip,
            "listener_personal_input": listener_3dmm_clip_personal,
        }

        [output_prior, output_decoder], regular_loss = (
            model(x=input_dict, p=listener_3dmm_clip_personal))
        # output_prior['encoded_prediction'].shape: [bs, k_appro==1, 1, 512]
        # output_prior['encoded_target'].shape: [bs, k_appro==1, 1, 512]
        # output_decoder['prediction_3dmm'].shape: [bs, k_appro==1, window_size, 58]
        # output_decoder['target_3dmm'].shape: [bs, k_appro==1, window_size, 58]

        output = criterion(output_prior, output_decoder)

        loss = output["loss"] + regular_loss  # whole training loss
        diff_prior_loss = output["encoded"]
        diff_decoder_loss = output["decoded"]

        iteration = batch_idx + len(train_loader) * epoch

        if writer is not None:
            writer.add_scalar("Train/whole_loss", loss.data.item(), iteration)
            writer.add_scalar("Train/diff_prior_loss", diff_prior_loss.data.item(), iteration)
            writer.add_scalar("Train/diff_decoder_loss", diff_decoder_loss.data.item(), iteration)
            writer.add_scalar("Train/regular_loss", regular_loss.data.item(), iteration)

        whole_losses.update(loss.data.item(), batch_size)
        diff_prior_losses.update(diff_prior_loss.data.item(), batch_size)
        diff_decoder_losses.update(diff_decoder_loss.data.item(), batch_size)
        # regular_losses.update(regular_loss.data.item(), batch_size)

        optimizer_mainnet.zero_grad()
        loss.backward()
        optimizer_hypernet.step()

    return whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg


def validate(model, val_loader, criterion, device):
    whole_losses = AverageMeter()
    diff_prior_losses = AverageMeter()
    diff_decoder_losses = AverageMeter()

    model.eval()
    for batch_idx, (
            speaker_audio_clip,
            _,  # speaker_video_clip
            speaker_emotion_clip,
            speaker_3dmm_clip,
            _,  # listener_video_clip
            listener_emotion_clip,
            listener_3dmm_clip,
            listener_3dmm_clip_personal,
            _,  # listener_reference
    ) in enumerate(tqdm(val_loader)):
        batch_size = speaker_audio_clip.shape[0]
        (speaker_audio_clip,  # (bs, token_len, 78)
         speaker_emotion_clip,  # (bs, token_len, 25)
         speaker_3dmm_clip,  # (bs, token_len, 58)
         listener_emotion_clip,  # (bs * k, token_len, 25)
         listener_3dmm_clip,  # (bs * k, token_len, 58)
         # (bs * k, token_len, 58)
         listener_3dmm_clip_personal) = \
            (speaker_audio_clip.to(device),
             speaker_emotion_clip.to(device),
             speaker_3dmm_clip.to(device),
             listener_emotion_clip.to(device),
             listener_3dmm_clip.to(device),
             listener_3dmm_clip_personal.to(device))

        with torch.no_grad():
            input_dict = {
                "speaker_audio": speaker_audio_clip,
                "speaker_emotion_input": speaker_emotion_clip,
                "speaker_3dmm_input": speaker_3dmm_clip,
                "listener_emotion_input": listener_emotion_clip,
                "listener_3dmm_input": listener_3dmm_clip,
                "listener_personal_input": listener_3dmm_clip_personal,
            }
            [output_prior, output_decoder], regular_loss = model(x=input_dict, p=listener_3dmm_clip_personal)

            output = criterion(output_prior, output_decoder)
            loss = output["loss"] + regular_loss  # whole training loss
            diff_prior_loss = output["encoded"]
            diff_decoder_loss = output["decoded"]

            whole_losses.update(loss.data.item(), batch_size)
            diff_prior_losses.update(diff_prior_loss.data.item(), batch_size)
            diff_decoder_losses.update(diff_decoder_loss.data.item(), batch_size)

    return whole_losses.avg, diff_prior_losses.avg, diff_decoder_losses.avg


def main(args):
    # load yaml config
    cfg = load_config(args=args, config_path=args.config)
    init_seed(seed=cfg.trainer.seed)  # seed initialization
    # lowest_val_loss = 10000

    # logging
    logging_path = get_logging_path(cfg.trainer.log_dir, exp_num=cfg.exp_num)
    os.makedirs(logging_path, exist_ok=True)
    logging.basicConfig(filename=logging_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    if cfg.writer:
        writer_path = get_tensorboard_path(cfg.trainer.tb_dir, exp_num=cfg.exp_num)
        writer = SummaryWriter(writer_path)
    else:
        writer = None

    train_loader = get_dataloader(cfg.dataset)
    # val_loader = get_dataloader(cfg.validation_dataset)

    # Set device ordinal if GPUs are available
    if torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')  # Adjust the device ordinal as needed
    else:
        device = torch.device('cpu')

    # our diffusion net
    diff_model = getattr(module_arch, cfg.trainer.model)(cfg, device)
    diff_model.to(device)

    # Main Net for modifying model weights.
    main_model = getattr(module_arch, cfg.main_model.type)(cfg, diff_model, device)
    main_model.to(device)

    # Optimizer for ModifierNet to modify the weight
    optimizer_hypernet = optim.SGD(params=main_model.hypernet.parameters(),
                                   lr=cfg.main_model.optimizer_hypernet.args.lr,
                                   momentum=cfg.main_model.optimizer_hypernet.args.momentum,
                                   weight_decay=cfg.main_model.optimizer_hypernet.args.weight_decay)

    # if cfg.main_model.args.resume is not None:  # resume optimizer
    #     optimizer_resume(cfg, net=main_model.hypernet, optimizer=optimizer_hypernet)

    optimizer_mainnet = optim.SGD(params=main_model.parameters(),
                                  lr=cfg.main_model.optimizer_mainnet.args.lr,
                                  momentum=cfg.main_model.optimizer_mainnet.args.momentum,
                                  weight_decay=cfg.main_model.optimizer_mainnet.args.weight_decay)

    criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)

    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.epochs):

        train_loss, diff_prior_loss, diff_decoder_loss = train(
            cfg, main_model, train_loader, optimizer_hypernet, optimizer_mainnet, criterion, epoch, writer, device
        )

        logging.info(
            "Epoch: {} train_whole_loss: {:.5f} diff_prior_loss: {:.5f} diff_decoder_loss: {:.5f}"
            .format(epoch + 1, train_loss, diff_prior_loss, diff_decoder_loss))

        if (epoch + 1) % cfg.trainer.val_period == 0:  # including the first epoch
            save_checkpoint(cfg, main_model.hypernet, optimizer_hypernet, (epoch + 1), is_best=False)


if __name__ == '__main__':
    main(args=parse_arg())
