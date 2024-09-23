import argparse
import os
from pathlib import Path
import time
from itertools import islice
from tqdm import tqdm
import re

import einops
import safetensors as st
import safetensors.torch
import torch
import torch.nn as nn
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration

from streamvc.model import StreamVC
from streamvc.train.discriminator import Discriminator
from streamvc.train.encoder_classifier import EncoderClassifier
from streamvc.train.libritts import get_libritts_dataloader
from streamvc.train.loss import GeneratorLoss, DiscriminatorLoss, FeatureLoss, ReconstructionLoss

accelerator = Accelerator(log_with="tensorboard",
                          project_config=ProjectConfiguration(
                              project_dir=os.getcwd(),
                              logging_dir=os.path.join(os.getcwd(), "logs")),
                          dataloader_config=DataLoaderConfiguration(split_batches=True))
# TODO: shouldn't we set it to True only if the input size is constant? - it errors without it
torch.backends.cudnn.benchmark = True
# TODO: isn't it enabled by default? we probably don't
# torch.backends.cudnn.enabled = True

NUM_CLASSES = 100
EMBEDDING_DIMS = 64
SAMPLES_PER_FRAME = 320
TRAIN_SPLIT = "train.clean.100"
DEV_SPLIT = "dev.clean"
TEST_SPLIT = "test.clean"
DEVICE = accelerator.device


def print_time(s):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    accelerator.print(f"[{current_time}] - {s}", flush=True)


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "K", "M", "G", "T", "P"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}{suffix}"
        num /= 1024.0


def print_cuda_memory(s):
    if accelerator.device.type != "cuda":
        print_time(s)
        return
    free, total = torch.cuda.mem_get_info()
    curr = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    size = {
        "allocated": curr,
        "total": total,
        "free": free,
        "peak": peak
    }

    print_time(
        " | ".join(
            map(lambda x: f"{x[0]} {sizeof_fmt(x[1]):8}", size.items()))
        + f" - {s}")


@torch.no_grad()
def get_batch_labels(hubert_model: nn.Module, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Get hubert output labels for a given audio samples batch.

    :param hubert_model: Hubert model with discrete output labels.
    :param batch: A batch of audio samples.
    :return: The output predictions generated by the Hubert model for the input batch.
    """
    labels = []
    frame_mask = mask.unfold(
        dimension=-1, size=SAMPLES_PER_FRAME, step=SAMPLES_PER_FRAME).all(dim=-1)
    for sample in batch:
        single_sample_batch = einops.rearrange(sample, 's -> 1 1 s')
        labels.append(hubert_model.units(single_sample_batch))
    labels = torch.stack(labels, dim=0)
    assert labels.shape == frame_mask.shape
    labels[~frame_mask] = -1
    return labels


@accelerator.on_main_process
def log_gradients(model, step):
    summary_writer = accelerator.get_tracker("tensorboard").tracker
    for name, param in model.named_parameters():
        if param.grad is not None:
            summary_writer.add_histogram(
                f"gradients/{name}", param.grad, global_step=step)


@accelerator.on_main_process
def log_labels(outputs_flat, labels_flat, step):
    _, predicted = torch.max(outputs_flat.data, 1)
    summary_writer = accelerator.get_tracker("tensorboard").tracker
    summary_writer.add_histogram(
        "labels/content_encoder", predicted, global_step=step)
    summary_writer.add_histogram(
        "labels/hubert", labels_flat, global_step=step)


def get_lr_Scheduler(optimizer, args, discriminator=False):
    if args.scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step,
            gamma=args.scheduler_gamma
        )
    elif args.scheduler == "LinearLR":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.scheduler_linear_start,
            end_factor=args.scheduler_linear_end,
            total_iters=args.num_epochs * args.limit_num_batches
        )
    elif args.scheduler == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=args.scheduler_gamma
        )
    elif args.scheduler == "OneCycleLR":
        max_lr = args.scheduler_onecycle_max
        if discriminator and args.lr_discriminator_multiplier is not None:
            max_lr = args.lr_discriminator_multiplier * max_lr
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=args.limit_num_batches,
            epochs=args.num_epochs,
            pct_start=args.scheduler_onecycle_pct_start,
            div_factor=args.scheduler_onecycle_div_factor,
            final_div_factor=args.scheduler_onecycle_final_div_factor
        )
    elif args.scheduler == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.scheduler_step,
            T_mult=1,
            eta_min=args.scheduler_cosine_eta_min
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def train_content_encoder(
    content_encoder: nn.Module, 
    hubert_model: nn.Module, 
    args: argparse.Namespace,
    starting_epoch: int = 0,
    starting_step: int = 0, 
) -> nn.Module:
    """
    Train a content encoder as a classifier to predict the same labels as a discrete hubert model.

    :param content_encoder: A content encoder wrapped with a linear layer to
    :param hubert_model: Hubert model with discrete output labels.
    :param lr: Learning rate.
    :param num_epochs: Number of epochs.
    :return: The trained content encoder wrapped with a linear layer for classification.
    """
    # TODO: add epochs or number of steps when we know how much time it takes to train the model.
    wrapped_content_encoder = EncoderClassifier(
        content_encoder, EMBEDDING_DIMS, NUM_CLASSES, dropout=args.encoder_dropout).train()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(
        params=wrapped_content_encoder.parameters(),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay
    )
    scheduler = get_lr_Scheduler(optimizer, args)

    dataloader = get_libritts_dataloader(
        TRAIN_SPLIT,
        args.batch_size,
        limit_samples=args.limit_batch_samples,
        streaming=args.dataset_streaming
    )
    dev_dataloader = get_libritts_dataloader(
        DEV_SPLIT,
        args.batch_size,
        limit_samples=args.limit_batch_samples,
        streaming=args.dataset_streaming
    )

    [
        wrapped_content_encoder,
        optimizer,
        dataloader,
        dev_dataloader,
        criterion,
        scheduler
    ] = accelerator.prepare(
        wrapped_content_encoder,
        optimizer,
        dataloader,
        dev_dataloader,
        criterion,
        scheduler
    )

    # TODO: distributed inference with the hubert model
    hubert_model.to(accelerator.device)
    costs = []
    global_step = 0
    for epoch in range(0, args.num_epochs):
        epoch += starting_epoch
        print_time(f"epoch num: {epoch}")
        for step, (batch, mask) in tqdm(enumerate(islice(dataloader, args.limit_num_batches))):
            step += starting_step
            labels = get_batch_labels(hubert_model, batch, mask)
            with accelerator.accumulate(wrapped_content_encoder):
                outputs = wrapped_content_encoder(batch)
                outputs_flat = outputs.view(-1, NUM_CLASSES)
                labels_flat = labels.view(-1)
                loss = criterion(outputs_flat, labels_flat)
                accelerator.backward(loss)

                if args.log_gradient_interval and (global_step + 1) % args.log_gradient_interval == 0:
                    log_gradients(wrapped_content_encoder, global_step)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step(global_step)
                accelerator.log(
                    {
                        "loss/content_encoder": loss.item(),
                        "lr/content_encoder": scheduler.get_last_lr()[0],
                        "allocated_memory": torch.cuda.max_memory_allocated()
                        if accelerator.device.type == "cuda"
                        else 0
                    },
                    step=global_step)
                costs.append(loss.item())

            # print loss
            if (global_step + 1) % args.log_interval == 0:
                print_time(
                    f'[{epoch}, {global_step:5}] loss: {torch.tensor(costs).mean().item():.4}')
                costs = []

            if args.log_labels_interval and (global_step + 1) % args.log_labels_interval == 0:
                log_labels(outputs_flat, labels_flat, global_step)

            # save model checkpoints
            if (global_step + 1) % args.model_checkpoint_interval == 0:
                accelerator.save_model(
                    wrapped_content_encoder,
                    save_directory=os.path.join(
                        args.checkpoint_path,
                        f"{args.run_name}_content_encoder_{epoch}_{step}"
                    ))

            if (global_step + 1) % args.accuracy_interval == 0:
                accuracy = compute_content_encoder_accuracy(
                    islice(dev_dataloader, 10),
                    wrapped_content_encoder,
                    hubert_model)
                accuracies = accelerator.gather_for_metrics([accuracy])
                accuracies = torch.tensor(accuracies)
                gathered_accuracy = accuracies.mean().item()
                accelerator.log(
                    {
                        "accuracy/content_encoder": gathered_accuracy
                    },
                    step=global_step)
                print_time(f"accuracy: {accuracy:.2f}%")
            if accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            global_step += 1
        starting_step = 0


@torch.no_grad()
def compute_content_encoder_accuracy(dataloader, wrapped_content_encoder: nn.Module, hubert_model: nn.Module):
    correct = 0
    total = 0
    wrapped_content_encoder.to(accelerator.device).eval()
    for (batch, mask) in dataloader:
        batch = batch.to(accelerator.device)
        labels = get_batch_labels(hubert_model, batch, mask)
        outputs = wrapped_content_encoder(batch)
        outputs_flat = outputs.view(-1, NUM_CLASSES)
        labels_flat = labels.view(-1)
        _, predicted = torch.max(outputs_flat.data, 1)
        total += torch.sum(labels_flat != -1).item()
        correct += (predicted == labels_flat).sum().item()
    wrapped_content_encoder.train()

    return 100 * correct / total


def train_streamvc(
    streamvc_model: StreamVC, 
    discriminator: Discriminator, 
    args: argparse.Namespace,
    starting_epoch: int = 0,
    starting_step: int = 0,
) -> None:
    """
       Trains a StreamVC model.

       :param streamvc_model: The model to train.
       :param args: Hyperparameters for training.
       """
    mem_limit = os.getenv('PYTORCH_LIMIT_PROCESS_MEM', None)

    #######################
    # Load PyTorch Models #
    #######################
    generator = streamvc_model
    

    for param in generator.content_encoder.parameters():
        param.requires_grad = False

    #####################
    # Create optimizers #
    #####################
    optimizer_generator = torch.optim.AdamW(
        params=[param for param in generator.parameters()
                if param.requires_grad],
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay)

    lr_discriminator = args.lr
    if args.lr_discriminator_multiplier is not None:
        lr_discriminator = args.lr_discriminator_multiplier * lr_discriminator
    optimizer_discriminator = torch.optim.AdamW(
        params=discriminator.parameters(),
        lr=lr_discriminator,
        betas=args.betas,
        weight_decay=args.weight_decay)

    scheduler_generator = get_lr_Scheduler(optimizer_generator, args)
    scheduler_discriminator = get_lr_Scheduler(
        optimizer_discriminator, args, discriminator=True)

    dataloader = get_libritts_dataloader(
        TRAIN_SPLIT,
        args.batch_size,
        limit_samples=args.limit_batch_samples,
        streaming=args.dataset_streaming
    )

    generator_loss_fn = GeneratorLoss()
    discriminator_loss_fn = DiscriminatorLoss()
    feature_loss_fn = FeatureLoss()
    reconstruction_loss_fn = ReconstructionLoss(
        gradient_checkpointing=args.gradient_checkpointing)

    [
        generator,
        discriminator,
        optimizer_generator,
        optimizer_discriminator,
        scheduler_generator,
        scheduler_discriminator,
        dataloader,
        generator_loss_fn,
        discriminator_loss_fn,
        feature_loss_fn,
        reconstruction_loss_fn
    ] = accelerator.prepare(
        generator,
        discriminator,
        optimizer_generator,
        optimizer_discriminator,
        scheduler_generator,
        scheduler_discriminator,
        dataloader,
        generator_loss_fn,
        discriminator_loss_fn,
        feature_loss_fn,
        reconstruction_loss_fn
    )

    costs = []
    global_step = 0 
    for epoch in range(0, args.num_epochs):
        epoch += starting_epoch
        print_time(f"epoch num: {epoch}")
        for step, (batch, mask) in enumerate(islice(dataloader, args.limit_num_batches)):
            step += starting_step
            x_pred_t = generator(batch, batch)
            # Remove the first 2 frames from the generated audio
            # because we match a output frame t with input frame t-2.
            x_pred_t = x_pred_t[..., SAMPLES_PER_FRAME * 2:]
            batch = batch[..., :x_pred_t.shape[-1]]

            mask_ratio = mask.sum(dim=-1) / mask.shape[-1]

            #######################
            # Train Discriminator #
            #######################

            discriminator.zero_grad()

            discriminator_fake_detached = discriminator(x_pred_t.detach())
            discriminator_real = discriminator(batch)

            discriminator_loss = discriminator_loss_fn(
                discriminator_real, discriminator_fake_detached, mask_ratio)

            accelerator.backward(discriminator_loss)

            if args.log_gradient_interval and (global_step + 1) % args.log_gradient_interval == 0:
                log_gradients(discriminator, global_step)

            optimizer_discriminator.step()
            scheduler_discriminator.step(global_step)

            ###################
            # Train Generator #
            ###################

            generator.zero_grad()

            discriminator_fake = discriminator(x_pred_t)

            # Compute adversarial loss.
            adversarial_loss = generator_loss_fn(
                discriminator_fake, mask_ratio)

            # Compute feature loss.
            feature_loss = feature_loss_fn(
                discriminator_real, discriminator_fake, mask_ratio)

            # Compute reconstruction loss.
            reconstruction_loss = reconstruction_loss_fn(
                batch, x_pred_t, mask_ratio)

            losses = (
                args.lambda_adversarial * adversarial_loss +
                args.lambda_feature * feature_loss +
                args.lambda_reconstruction * reconstruction_loss)

            accelerator.backward(losses)

            if args.log_gradient_interval and (global_step + 1) % args.log_gradient_interval == 0:
                log_gradients(generator, global_step)

            optimizer_generator.step()
            scheduler_generator.step(global_step)

            ######################
            # Update tensorboard #
            ######################
            costs.append([
                discriminator_loss.item(),
                adversarial_loss.item(),
                feature_loss.item(),
                reconstruction_loss.item()
            ])

            accelerator.log(
                {
                    "loss/discriminator": discriminator_loss.item(),
                    "loss/adversarial": adversarial_loss.item(),
                    "loss/feature_matching": feature_loss.item(),
                    "loss/reconstruction": reconstruction_loss.item(),
                    "lr/generator": scheduler_generator.get_last_lr()[0],
                    "lr/discriminator": scheduler_discriminator.get_last_lr()[0],
                    "allocated_memory": torch.cuda.max_memory_allocated()
                    if accelerator.device.type == "cuda"
                    else 0
                },
                step=global_step)

            if (global_step + 1) % args.log_interval == 0:
                print_time(
                    f'[{epoch}, {step:5}] loss: {torch.tensor(costs).mean().item():.4}')
                costs = []
            if (global_step + 1) % args.model_checkpoint_interval == 0:
                accelerator.save_model(
                    generator,
                    save_directory=os.path.join(
                        args.checkpoint_path,
                        f"{args.run_name}_generator_{epoch}_{step}"
                    ))
                accelerator.save_model(
                    discriminator,
                    save_directory=os.path.join(
                        args.checkpoint_path,
                        f"{args.run_name}_discriminator_{epoch}_{step}"
                    ))
            if accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            global_step += 1
        starting_step = 0


def main(args):
    """Main function for training StreamVC model."""
    print_time(f"DEVICE={accelerator.device}")
    hps = dict(vars(args))
    hps["num processes"] = accelerator.num_processes
    hps["mixed precision"] = accelerator.mixed_precision
    hps["gradient accumulation steps"] = accelerator.gradient_accumulation_steps

    if accelerator.gradient_accumulation_steps > 1 and args.module_to_train != "content-encoder":
        raise ValueError(
            "Gradient accumulation is not supported for the decoder training.")

    iter_keys = [key for key, value in hps.items()
                 if isinstance(value, (list, tuple))]
    for key in iter_keys:
        for i, value in enumerate(hps[key]):
            hps[f"{key}[{i}]"] = value
        del hps[key]

    bad_keys = [key for key, value in hps.items()
                if not isinstance(value, (int, float, str, bool))]

    for key in bad_keys:
        del hps[key]

    print_time(f"{hps=}")
    print_time(f"module-to-train: {args.module_to_train}")

    ## Load Models With Checkpoints
    latest_encoder_step = -1
    latest_encoder_epoch = -1

    encoder_pattern = r"{}_content_encoder_(\d+)_(\d+)".format(args.run_name)
    
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
    for dirname in os.listdir(args.checkpoint_path):
        re_match = re.search(encoder_pattern, dirname)
        if not re_match:
            continue
        epoch = int(re_match.group(1))
        step = int(re_match.group(2))
        if epoch < latest_encoder_epoch or (epoch == latest_encoder_epoch and step < latest_encoder_step):
            continue
        latest_encoder_epoch = epoch
        latest_encoder_step = step

    accelerator.init_trackers(args.run_name, config=hps)
    streamvc = StreamVC(
        gradient_checkpointing=args.gradient_checkpointing)

    if latest_encoder_epoch != -1:
        encoder_checkpoint = "{}/streamvc_content_encoder_{}_{}/model.safetensors".format(
            args.checkpoint_path,
            latest_encoder_epoch,
            latest_encoder_step
        )
        wrapped_encoder_state_dict = st.torch.load_file(
            encoder_checkpoint)
        encoder_state_dict = {
            key[len("encoder."):]: value
            for key, value in wrapped_encoder_state_dict.items()
            if key.startswith("encoder.")
        }
        streamvc.content_encoder.load_state_dict(encoder_state_dict)

        print("Loaded checkpoint from: ", encoder_checkpoint)
    else:
        latest_encoder_epoch = 0
        latest_encoder_step = 0
    if args.module_to_train in ["content-encoder", "all"]:
        content_encoder = streamvc.content_encoder
        hubert_model = torch.hub.load("bshall/hubert:main", "hubert_discrete",
                                      trust_repo=True).to(torch.float32).eval()
        print_time(f"Training encoder. Starting step: {latest_encoder_step}")

        starting_step = latest_encoder_step + 1
        starting_epoch = max(0, latest_encoder_epoch)

        train_content_encoder(
            content_encoder, hubert_model, args, starting_epoch=starting_epoch, starting_step=starting_step)    
    
    latest_discriminator_step = -1
    latest_discriminator_epoch = -1
    latest_generator_step = -1
    latest_generator_epoch = -1
    discriminator_pattern = r"{}_discriminator_(\d+)_(\d+)".format(args.run_name)
    generator_pattern = r"{}_generator_(\d+)_(\d+)".format(args.run_name)
    for dirname in os.listdir(args.checkpoint_path):
        re_match = re.search(discriminator_pattern, dirname)
        if re_match:
            epoch = int(re_match.group(1))
            step = int(re_match.group(2))
            if epoch < latest_discriminator_epoch or (epoch == latest_discriminator_epoch and step < latest_discriminator_step):
                continue
            latest_discriminator_epoch = epoch
            latest_discriminator_step = step
        re_match = re.search(generator_pattern, dirname)
        if re_match:
            epoch = int(re_match.group(1))
            step = int(re_match.group(2))
            if epoch < latest_generator_epoch or (epoch == latest_generator_epoch and step < latest_generator_step):
                continue
            latest_generator_epoch = epoch
            latest_generator_step = step
    
    if args.module_to_train in ["decoder-and-speaker", "all"]:
        discriminator = discriminator = Discriminator(
                gradient_checkpointing=args.gradient_checkpointing)
        if latest_discriminator_epoch != -1:
            discriminator_checkpoint = "{}/{}_discriminator_{}_{}/model.safetensors".format(
                args.checkpoint_path,
                args.run_name,
                latest_discriminator_epoch,
                latest_discriminator_step
            )
            print(discriminator_checkpoint)
            discriminator.load_state_dict(
                st.torch.load_file(discriminator_checkpoint))
            print("Loaded checkpoint from: ", discriminator_checkpoint)
        
        # only load the previous generator checkpoint if the encoder was not changed
        if latest_generator_epoch != -1 and args.module_to_train == "decoder-and-speaker":
            generator_checkpoint = "{}/{}_generator_{}_{}/model.safetensors".format(
                args.checkpoint_path,
                args.run_name,
                latest_generator_epoch,
                latest_generator_step
            )
            streamvc.load_state_dict(
                st.torch.load_file(generator_checkpoint))
            print("Loaded checkpoint from: ", generator_checkpoint)
        
        starting_step = latest_generator_step + 1
        starting_epoch = max(latest_generator_epoch, 0)

        print_time(f"Training decoder. Starting step: {starting_step}")
        train_streamvc(streamvc, discriminator, args, starting_epoch=starting_epoch, starting_step=starting_step)

    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for the StreamVC model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General settings.
    parser.add_argument("--run-name", type=str, default="streamvc",
                        help="Name of the training run for identification purposes.")
    parser.add_argument("--module-to-train", type=str,
                        choices=["content-encoder",
                                 "decoder-and-speaker", "all"],
                        default="all",
                        help="Specify which module to train: 'content-encoder', 'decoder-and-speaker', or 'all' "
                             "for both.")
    parser.add_argument("--content-encoder-checkpoint", type=str, default="",
                        help="Path to the content encoder checkpoint. Must be provided when --model-to-train is "
                             "decoder-and-speaker")
    parser.add_argument("--no-dataset-streaming", action="store_false", dest="dataset_streaming",
                        help="Download the dataset to disk instead of streaming it.")

    # General hyperparameters.
    parser.add_argument("--batch-size", type=int, default=24,
                        help="Batch size for training.")
    parser.add_argument("--limit-num-batches", type=int, default=None,
                        help="Limit the number of batches per epoch. Use None for no limit.")
    parser.add_argument("--limit-batch-samples", type=int, default=16_000 * 10,
                        help="Limit the number of samples for audio signal in the batch.")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=4e-3,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.5, 0.9),
                        help="Beta parameters for the Adam or AdamW optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="Weight decay for the optimizer.")
    parser.add_argument("--no-gradient-checkpointing", action="store_false",
                        dest='gradient_checkpointing', default=True,
                        help="Disable gradient checkpointing to increase compute speed at the cost of increased memory"
                             "usage.")
    # LR schedualers
    parser.add_argument("--scheduler", type=str, default="StepLR",
                        choices=["StepLR", "LinearLR",
                                 "ExponentialLR", "OneCycleLR",
                                 "CosineAnnealingWarmRestarts"],
                        help="Learning rate scheduler to use.")
    parser.add_argument("--scheduler-step", type=int, default=100,
                        help="Step interval for StepLR learning rate scheduler updates. Or T_0 for CosineAnnealingWarmRestarts.")
    parser.add_argument("--scheduler-gamma", type=float, default=0.1,
                        help="Gamma parameter for StepLR, ExponentialLR learning rate schedulers, controlling the decay rate.")
    parser.add_argument("--scheduler-linear-start", type=float, default=1.0,
                        help="Initial learning rate mutiplier for LinearLR learning rate scheduler.")
    parser.add_argument("--scheduler-linear-end", type=float, default=1.0,
                        help="Final learning rate mutiplier for LinearLR learning rate scheduler.")
    parser.add_argument("--scheduler-onecycle-max", type=float, default=1e-3,
                        help="Max learning rate for OneCycleLR learning rate scheduler.")
    parser.add_argument("--scheduler-onecycle-pct-start", type=float, default=0.3,
                        help="The percentage of the cycle spent increasing the learning rate "
                        + "for OneCycleLR learning rate scheduler.")
    parser.add_argument("--scheduler-onecycle-div-factor", type=float, default=25,
                        help="Determines the initial learning rate via initial_lr = max_lr/div_factor "
                        + "for OneCycleLR learning rate scheduler.")
    parser.add_argument("--scheduler-onecycle-final-div-factor", type=float, default=1e4,
                        help="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor "
                        + "for OneCycleLR learning rate scheduler.")
    parser.add_argument("--scheduler-cosine-eta-min", type=float, default=0,
                        help="Minimum learning rate for CosineAnnealingWarmRestarts learning rate scheduler.")

    # Content encoder hyperparameters.
    parser.add_argument("--encoder-dropout", type=float, default=0.1,
                        help="Dropout rate for the content encoder training.")

    # Decoder hyperparameters.
    parser.add_argument("--lambda-feature", type=float, default=100,
                        help="Weight of the feature matching loss.")
    parser.add_argument("--lambda-reconstruction", type=float, default=1,
                        help="Weight of the reconstruction loss.")
    parser.add_argument("--lambda-adversarial", type=float, default=1,
                        help="Weight of the adversarial loss.")
    parser.add_argument("--lr-discriminator-multiplier", type=float, default=None,
                        help="Learning rate multiplier for the discriminator, if None than lr is same as generator.")
    # Logs and outputs.
    parser.add_argument("--model-checkpoint-interval", type=int, default=100,
                        help="Interval (in steps) at which to save model checkpoints.")
    parser.add_argument("--accuracy-interval", type=int, default=100,
                        help="Interval (in steps) at which to compute and log accuracy.")
    parser.add_argument("--log-interval", type=int, default=20,
                        help="Interval (in steps) at which to log training metrics.")
    parser.add_argument("--log-gradient-interval", type=int, default=None,
                        help="Interval (in steps) at which to log gradient information. Use None to disable.")
    parser.add_argument("--log-labels-interval", type=int, default=None,
                        help="Interval (in steps) at which to log label information. Use None to disable.")
    parser.add_argument("--checkpoint-path", type=str,
                        default=os.path.join(os.environ.get(
                            "HF_HOME", os.getcwd()), "checkpoints"),
                        help="Path to save model checkpoints.")

    args = parser.parse_args()

    # if args.module_to_train == "decoder-and-speaker":
    #     assert args.content_encoder_checkpoint, "content-encoder-checkpoint is required for decoder training"

    main(args)
