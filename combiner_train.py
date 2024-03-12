from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import clip
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.data_utils import base_path, squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from utils.utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, \
    extract_index_features, generate_randomized_fiq_caption, device, ShowBest, ShowBestCIRR
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics
from modeling import train_models_map
import pdb


def combiner_training_fiq(args, train_dress_types: List[str], val_dress_types: List[str],
                          projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                          combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int,
                          transform: str, save_training: bool, save_best: bool, **kwargs):

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/combiner_trained_on_fiq_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": args.maple_n_ctx}
    args.design_details = design_details

    if args.network in train_models_map.keys():
        model = train_models_map[args.network](args)
    else:
        raise NotImplementedError(f"Unkonw model types: {args.network}")

    model = model.to(device)
    model.eval().float()

    input_dim = model.clip.visual.input_resolution
    clip_preprocess = model.clip_preprocess

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    if kwargs.get("clip_model_path"):
        print('Trying to load the CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        saved_state_dict = torch.load(clip_model_path, map_location=device)
        model.clip.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.model_s1_path is not None:
        print("Trying to load the model")
        ckpt = torch.load(args.model_s1_path, map_location=device)
        if args.network == 'clip4cir_maple_final_s2':
            model.load_state_dict(ckpt["CLIP4cirMaPLeFinalS1"], strict=False)
        else:
            model.load_state_dict(ckpt["CLIP4cirMaPLe"], strict=False)
        print('model loaded successfully')

    if args.img2txt_model_path:
        print('Trying to load the pic2word model')
        ckpt = torch.load(args.img2txt_model_path, map_location=device)
        model.prompt_learner.img2token.load_state_dict(ckpt[list(ckpt.keys())[-1]])
        print("img2txt model loaded successfully")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    tar_indexfeats_list = []
    index_names_list = []

    # Define the validation datasets and extract the validation index features for each dress_type
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
        index_features_and_names = extract_index_features(args, classic_val_dataset, model)
        tar_indexfeats_list.append(index_features_and_names[0])
        index_names_list.append(index_features_and_names[1])

    # Define the train dataset
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    if args.optimizer == 'combiner_aligner':
        optimizer = optim.Adam([
            {'params': model.combiner.parameters(), 'lr': combiner_lr},
            {'params': model.aligner.parameters(), 'lr': combiner_lr}
        ])
    elif args.optimizer == 'combiner_prompt_learner':
        optimizer = optim.Adam([
            {'params': model.combiner.parameters(), 'lr': combiner_lr},
            {'params': model.prompt_learner.parameters(), 'lr': 2e-6}
        ])
    elif args.optimizer == 'combiner':
        optimizer = optim.Adam(model.combiner.parameters(), lr=combiner_lr)
    else:
        raise NotImplementedError(f"Unknown type {args.optimizer}")
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    show_best = ShowBest()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        # if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
        #     clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory

        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        if args.optimizer == 'combiner_aligner':
            model.combiner.train()
            model.aligner.train()
        elif args.optimizer == 'combiner_prompt_learner':
            model.combiner.train()
            model.prompt_learner.train()
        elif args.optimizer == 'combiner':
            model.combiner.train()
        else:
            raise NotImplementedError(f"unknown type {args.optimizer}")
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
            step = len(train_bar) * epoch + idx
            images_in_batch = reference_images.size(0)

            optimizer.zero_grad()
            show_best(epoch, best_avg_recall)

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
            flattened_captions: list = np.array(captions).T.flatten().tolist()
            input_captions = generate_randomized_fiq_caption(flattened_captions)
            text_inputs = clip.tokenize(input_captions, truncate=True).to(device, non_blocking=True)

            logits = model(reference_images, text_inputs, target_images)
            ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
            loss = crossentropy_criterion(logits, ground_truth)

            # Backprogate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            update_train_running_results(train_running_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

        train_epoch_loss = float(
            train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])


        # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            # model = model.float()  # In validation we use fp32 CLIP model
            if args.optimizer == 'combiner_aligner':
                model.combiner.eval()
                model.aligner.eval()
            elif args.optimizer == 'combiner_prompt_learner':
                model.combiner.eval()
                model.prompt_learner.eval()
            elif args.optimizer == 'combiner':
                model.combiner.eval()
            else:
                raise NotImplementedError(f"Unkonw type {args.optimizer}")
            recalls_at10 = []
            recalls_at50 = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            for relative_val_dataset, tar_indexfeats, index_names, idx in zip(relative_val_datasets,
                                                                              tar_indexfeats_list,
                                                                              index_names_list,
                                                                              idx_to_dress_mapping):
                recall_at10, recall_at50 = compute_fiq_val_metrics(args, relative_val_dataset, model, 
                                                                   tar_indexfeats, index_names)
                recalls_at10.append(recall_at10)
                recalls_at50.append(recall_at50)

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
            })

            print(json.dumps(results_dict, indent=4))


            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # Save model
            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    # if args.aligner:
                    #     save_model('combiner', epoch, model.aligner, training_path)
                    save_model('combiner', epoch, model.combiner, training_path)
                    save_model('model', epoch, model, training_path)
                elif not save_best:
                    save_model(f'combiner_{epoch}', epoch, model.combiner, training_path)


def combiner_training_cirr(args, projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                           combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int, transform: str,
                           save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on CIRR dataset keeping frozen the CLIP model
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If you want to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/combiner_trained_on_cirr_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": args.maple_n_ctx}
    args.design_details = design_details

    if args.network in train_models_map.keys():
        model = train_models_map[args.network](args)
    else:
        raise NotImplementedError(f"Unkonw model types: {args.network}")

    model = model.to(device)
    model.eval().float()

    input_dim = model.clip.visual.input_resolution
    clip_preprocess = model.clip_preprocess

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    if kwargs.get("clip_model_path"):
        print('Trying to load the fine-tuned CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        state_dict = torch.load(clip_model_path, map_location=device)
        model.load_state_dict(state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.model_s1_path is not None:
        print("Trying to load the model")
        ckpt = torch.load(args.model_s1_path, map_location=device)
        if args.network == 'clip4cir_maple_final_s2':
            model.load_state_dict(ckpt["CLIP4cirMaPLeFinalS1"], strict=False)
        else:
            model.load_state_dict(ckpt["CLIP4cirMaPLe"], strict=False)
        print('model loaded successfully')

    if args.img2txt_model_path:
        print('Trying to load the pic2word model')
        ckpt = torch.load(args.img2txt_model_path, map_location=device)
        model.prompt_learner.img2token.load_state_dict(ckpt[list(ckpt.keys())[-1]])
        print("img2txt model loaded successfully")

    # Define the validation datasets and extract the validation index features
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    val_index_features, val_index_names = extract_index_features(args, classic_val_dataset, model)

    # Define the combiner and the train dataset
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size, num_workers=8,
                                       pin_memory=True, collate_fn=collate_fn, drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    if args.optimizer == 'combiner_aligner':
        optimizer = optim.Adam([
            {'params': model.combiner.parameters(), 'lr': combiner_lr},
            {'params': model.aligner.parameters(), 'lr': combiner_lr}
        ])
    elif args.optimizer == 'combiner_prompt_learner':
        optimizer = optim.Adam([
            {'params': model.combiner.parameters(), 'lr': combiner_lr},
            {'params': model.prompt_learner.parameters(), 'lr': 2e-6}
        ])
    elif args.optimizer == 'combiner':
        optimizer = optim.Adam(model.combiner.parameters(), lr=combiner_lr)
    else:
        raise NotImplementedError(f"Unknown type {args.optimizer}")
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    show_best = ShowBestCIRR()

    # When save_best == True initialize the best results to zero
    if save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0
    best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        # if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
        #     clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory
        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        if args.optimizer == 'combiner_aligner':
            model.combiner.train()
            model.aligner.train()
        elif args.optimizer == 'combiner_prompt_learner':
            model.combiner.train()
            model.prompt_learner.train()
        elif args.optimizer == 'combiner':
            model.combiner.train()
        else:
            raise NotImplementedError(f"unknown type {args.optimizer}")
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()
            show_best(epoch, best_avg_recall, best_harmonic, best_geometric, best_arithmetic)

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            text_inputs = clip.tokenize(captions, truncate=True).to(device, non_blocking=True)

            logits = model(reference_images, text_inputs, target_images)
            ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
            loss = crossentropy_criterion(logits, ground_truth)

            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_train_running_results(train_running_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

        train_epoch_loss = float(
            train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])


        # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            # clip_model = clip_model.float()  # In validation we use fp32 CLIP model
            if args.optimizer == 'combiner_aligner':
                model.combiner.eval()
                model.aligner.eval()
            elif args.optimizer == 'combiner_prompt_learner':
                model.combiner.eval()
                model.prompt_learner.eval()
            elif args.optimizer == 'combiner':
                model.combiner.eval()
            else:
                raise NotImplementedError(f"Unkonw type {args.optimizer}")

            # Compute and log validation metrics
            results = compute_cirr_val_metrics(args, relative_val_dataset, model, val_index_features, val_index_names)
            group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

            results_dict = {
                'group_recall_at1': group_recall_at1,
                'group_recall_at2': group_recall_at2,
                'group_recall_at3': group_recall_at3,
                'recall_at1': recall_at1,
                'recall_at5': recall_at5,
                'recall_at10': recall_at10,
                'recall_at50': recall_at50,
                'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                'arithmetic_mean': mean(results),
                'harmonic_mean': harmonic_mean(results),
                'geometric_mean': geometric_mean(results)
            }

            print(json.dumps(results_dict, indent=4))

            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # Save model
            if save_training:
                if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                    best_arithmetic = results_dict['arithmetic_mean']
                    save_model('combiner_arithmetic', epoch, model.combiner, training_path)
                    save_model('model_arithmetic', epoch, model, training_path)
                if save_best and results_dict['harmonic_mean'] > best_harmonic:
                    best_harmonic = results_dict['harmonic_mean']
                    save_model('combiner_harmonic', epoch, model.combiner, training_path)
                    save_model('model_harmonic', epoch, model, training_path)
                if save_best and results_dict['geometric_mean'] > best_geometric:
                    best_geometric = results_dict['geometric_mean']
                    save_model('combiner_geometric', epoch, model.combiner, training_path)
                    save_model('model_geometric', epoch, model, training_path)
                if save_best and results_dict['mean(R@5+R_s@1)'] > best_avg_recall:
                    best_avg_recall = results_dict['mean(R@5+R_s@1)']
                    save_model('combiner_avg', epoch, model.combiner, training_path)
                    save_model('model_avg', epoch, model, training_path)
                if not save_best:
                    save_model(f'combiner_{epoch}', epoch, model.combiner, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--combiner-lr", default=2e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int, help="Batch size of the Combiner training")
    parser.add_argument("--clip-bs", default=32, type=int, help="Batch size during CLIP feature extraction")
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--asynchronous", default=False, action="store_true")
    parser.add_argument("--network", default='clip4cir_maple_s2', type=str)
    parser.add_argument("--model-s1-path", type=str)
    parser.add_argument(
        "--maple-n-ctx", type=int, default=3, help=""
    )
    parser.add_argument(
        "--maple-ctx-init", type=str, default="a photo of", help=""
    )
    parser.add_argument(
        "--maple-prompt-depth", type=int, default=9, help=""
    )
    parser.add_argument(
        "--input-size", type=int, default=224, help=""
    )
    parser.add_argument(
        "--combiner", default='Combiner', type=str
    )
    parser.add_argument(
        "--final", default=False, action='store_true'
    )
    parser.add_argument(
        "--num-workers", default=8, type=int
    )
    parser.add_argument(
        "--sum-combiner", default=False, action='store_true'
    )
    parser.add_argument(
        "--fixed-image-encoder", default=False, action='store_true'
    )
    parser.add_argument(
        "--img2txt-model-path", default='', type=str
    )
    parser.add_argument(
        "--embed-size", default=512, type=int
    )
    parser.add_argument(
        "--aligner", default=False, action='store_true'
    )
    parser.add_argument(
        "--cross-attn-layer", default=4, type=int
    )
    parser.add_argument(
        "--cross-attn-head", default=2, type=int
    )
    parser.add_argument(
        "--bsc-loss", default=False, action='store_true'
    )
    parser.add_argument(
        "--txt2img", default=False, action='store_true'
    )
    parser.add_argument(
        "--optimizer", default='combiner', type=str
    )
    parser.add_argument(
        "--router", default=False, action='store_true'
    )
    parser.add_argument(
        "--mu", default=0.1, type=float
    )

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "clip_model_path": args.clip_model_path,
        "combiner_lr": args.combiner_lr,
        "batch_size": args.batch_size,
        "clip_bs": args.clip_bs,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "asynchronous": args.asynchronous,
        "model_s1_path": args.model_s1_path,
        "maple_ctx_init": args.maple_ctx_init,
        "maple_n_ctx": args.maple_n_ctx,
        "maple_prompt_depth": args.maple_prompt_depth,
        "combiner": args.combiner,
        "final": args.final,
        "num_workers": args.num_workers,
        "fixed_image_encoder": args.fixed_image_encoder,
        "img2txt_model_path": args.img2txt_model_path,
        "embed_size": args.embed_size,
        "aligner": args.aligner,
        "cross_attn_layer": args.cross_attn_layer,
        "cross_attn_head": args.cross_attn_head,
        "bsc_loss": args.bsc_loss,
        "txt2img": args.txt2img,
        "optimizer": args.optimizer,
        "router": args.router,
        "mu": args.mu
    }

    if args.dataset.lower() == 'cirr':
        combiner_training_cirr(args, **training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        combiner_training_fiq(args, **training_hyper_params)
