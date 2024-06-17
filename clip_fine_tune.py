# from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
from clip import clip

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
# from combiner import Combiner4ft
from utils.utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device, ShowBest, ShowBestCIRR
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics
from modeling import train_models_map
import pdb


def clip_finetune_fiq(args, train_dress_types: List[str], val_dress_types: List[str],
                      num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, encoder: str, save_best: bool,
                      **kwargs):

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_fiq_{clip_model_name}_{training_start}")
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

    if args.img2txt_model_path:
        print("Trying to load the modal")
        ckpt = torch.load(args.img2txt_model_path, map_location=device)
        model.prompt_learner.img2token.load_state_dict(ckpt[list(ckpt.keys())[-1]])
        print("img2txt model loaded successfully.")

        for param in model.fixed_image_encoder.parameters():
            param.requires_grad = False

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        ckpt = torch.load(args.clip_model_path, map_location=device)
        model.clip.load_state_dict(ckpt[list(ckpt.keys())[-1]])
        print('CLIP model loaded successfully')

    if args.clip_image_encoder_path:
        print('Trying to load the CLIP image encoder model')
        ckpt = torch.load(args.clip_image_encoder_path, map_location=device)
        model.clip_image_encoder.load_state_dict(ckpt[list(ckpt.keys())[-1]])
        print("clip image encoder loaded successfully.")

            
    if args.asynchronous:
        if encoder == 'text':
            print('Only the CLIP text encoder will be fine-tuned')
            for param in model.text_encoder.parameters():
                param.requires_grad = True
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.clip_image_encoder.parameters():
                param.requires_grad = False

        elif encoder == 'target':
            print('Only the CLIP target image encoder will be fine-tuned')
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.clip_image_encoder.parameters():
                param.requires_grad = True

        elif encoder == 'text_target':
            print('Both CLIP text and target image encoders will be fine-tuned')
            for param in model.text_encoder.parameters():
                param.requires_grad = True
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.clip_image_encoder.parameters():
                param.requires_grad = True
        
        elif encoder == 'reference_text_target':
            print('All encoders will be fine-tuned')

        elif encoder == 'none':
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.clip_image_encoder.parameters():
                param.requires_grad = False
            
        else:
            raise ValueError("encoder parameter should be in ['text', 'image', both']")
    else:
        if encoder == 'text':
            print('Only the CLIP text encoder will be fine-tuned')
            for param in model.image_encoder.parameters():
                param.requires_grad = False

        elif encoder == 'image':
            print('Only the CLIP target image encoder will be fine-tuned')
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            for param in model.clip.parameters():
                param.requires_grad = False
            for param in model.image_encoder.parameters():
                param.requires_grad = True
        
        elif encoder == 'both':
            print('All encoders will be fine-tuned')
            
        else:
            raise ValueError("encoder parameter should be in ['text', 'image', both']")

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


    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs
    if encoder == 'text':
        index_features_list = []
        index_names_list = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)

        if encoder == 'text':
            index_features_and_names = extract_index_features(args, classic_val_dataset, model)
            index_features_list.append(index_features_and_names[0])
            index_names_list.append(index_features_and_names[1])

    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=args.num_workers, pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)


    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
    # optimizer  = optim.AdamW([
    #     {'params': filter(lambda p: p.requires_grad, model.prompt_learner.parameters()), 'lr': learning_rate,
    #      'betas': (0.9, 0.999), 'eps': 1e-7}
    # ])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0
    show_best = ShowBest()

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        model.train()
        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()
            show_best(epoch, best_avg_recall)

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            
            # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
            flattened_captions: list = np.array(captions).T.flatten().tolist()
            captions = generate_randomized_fiq_caption(flattened_captions)
            text_inputs = clip.tokenize(captions, context_length=77, truncate=True).to(device, non_blocking=True) # [B, 77]

            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                # logits = model(reference_images, text_inputs, target_images)
                # ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                # loss = crossentropy_criterion(logits, ground_truth)
                loss = model(reference_images, text_inputs, target_images)
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
            model.eval()
            recalls_at10 = []
            recalls_at50 = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                        idx_to_dress_mapping):
                if encoder == 'text':
                    tar_indexfeats, index_names = index_features_list[idx], index_names_list[idx]
                else:
                    tar_indexfeats, index_names = extract_index_features(args, classic_val_dataset, model)

                recall_at10, recall_at50 = compute_fiq_val_metrics(args, relative_val_dataset, model, tar_indexfeats, index_names)
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

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('best_model', epoch, model, training_path)
                elif not save_best:
                    save_model(f'checkpoint_{epoch}', epoch, model, training_path)


def clip_finetune_cirr(args, num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool, encoder: str, save_best: bool,
                       **kwargs):
    """
    Fine-tune CLIP on the CIRR dataset using as combining function the image-text element-wise sum
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    :return:
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_cirr_{clip_model_name}_{training_start}")
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

    if args.img2txt_model_path:
        print("Trying to load the modal")
        ckpt = torch.load(args.img2txt_model_path, map_location=device)
        model.prompt_learner.img2token.load_state_dict(ckpt[list(ckpt.keys())[-1]])
        print("img2txt model loaded successfully.")

        for param in model.fixed_image_encoder.parameters():
            param.requires_grad = False

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        ckpt = torch.load(args.clip_model_path, map_location=device)
        model.clip.load_state_dict(ckpt[list(ckpt.keys())[-1]])
        print('CLIP model loaded successfully')

    if args.clip_image_encoder_path:
        print('Trying to load the CLIP image encoder model')
        ckpt = torch.load(args.clip_image_encoder_path, map_location=device)
        model.clip_image_encoder.load_state_dict(ckpt[list(ckpt.keys())[-1]])
        print("clip image encoder loaded successfully.")

    if args.asynchronous:
        if encoder == 'text':
            print('Only the CLIP text encoder will be fine-tuned')
            for param in model.text_encoder.parameters():
                param.requires_grad = True
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.clip_image_encoder.parameters():
                param.requires_grad = False

        elif encoder == 'target':
            print('Only the CLIP target image encoder will be fine-tuned')
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.clip_image_encoder.parameters():
                param.requires_grad = True

        elif encoder == 'text_target':
            print('Both CLIP text and target image encoders will be fine-tuned')
            for param in model.text_encoder.parameters():
                param.requires_grad = True
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.clip_image_encoder.parameters():
                param.requires_grad = True
        
        elif encoder == 'reference_text_target':
            print('All encoders will be fine-tuned')

        elif encoder == 'none':
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            for param in model.image_encoder.parameters():
                param.requires_grad = False
            for param in model.clip_image_encoder.parameters():
                param.requires_grad = False
            
        else:
            raise ValueError("encoder parameter should be in ['text', 'image', both']")
    else:
        if encoder == 'text':
            print('Only the CLIP text encoder will be fine-tuned')
            for param in model.image_encoder.parameters():
                param.requires_grad = False

        elif encoder == 'image':
            print('Only the CLIP target image encoder will be fine-tuned')
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            for param in model.clip.parameters():
                param.requires_grad = False
            for param in model.image_encoder.parameters():
                param.requires_grad = True
        
        elif encoder == 'both':
            print('All encoders will be fine-tuned')
            
        else:
            raise ValueError("encoder parameter should be in ['text', 'image', both']")

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

    # Define the validation datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs
    if encoder == 'text':
        val_index_features, val_index_names = extract_index_features(args, classic_val_dataset, model)

    # Define the train dataset and the combining function
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=args.num_workers, pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)
    combining_function = element_wise_sum
    # combining_function = Combiner4ft(feature_dim).to(device, non_blocking=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
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
    # start with the training loop
    print("Training loop started")
    for epoch in range(num_epochs):
        model.train()       
        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()
            show_best(epoch, best_avg_recall, best_harmonic, best_geometric, best_arithmetic)

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            text_inputs = clip.tokenize(captions, context_length=77, truncate=True).to(device, non_blocking=True)

            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                # logits = model(reference_images, text_inputs, target_images)
                # ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                # loss = crossentropy_criterion(logits, ground_truth)
                loss = model(reference_images, text_inputs, target_images)

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
            model.eval()
            if encoder != 'text':
                val_index_features, val_index_names = extract_index_features(args, classic_val_dataset, model)
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

            if save_training:
                if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                    best_arithmetic = results_dict['arithmetic_mean']
                    save_model('tuned_clip_arithmetic', epoch, model, training_path)
                if save_best and results_dict['harmonic_mean'] > best_harmonic:
                    best_harmonic = results_dict['harmonic_mean']
                    save_model('tuned_clip_harmonic', epoch, model, training_path)
                if save_best and results_dict['geometric_mean'] > best_geometric:
                    best_geometric = results_dict['geometric_mean']
                    save_model('tuned_clip_geometric', epoch, model, training_path)
                if save_best and results_dict['mean(R@5+R_s@1)'] > best_avg_recall:
                    best_avg_recall = results_dict['mean(R@5+R_s@1)']
                    save_model('tuned_clip_best', epoch, model, training_path)
                if not save_best:
                    save_model(f'tuned_clip_{epoch}', epoch, model, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument(
        "--api-key", type=str, help="api for Comet logging")
    parser.add_argument(
        "--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument(
        "--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument(
        "--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument(
        "--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument(
        "--encoder", default='both', type=str,
                        help="Which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']")
    parser.add_argument(
        "--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument(
        "--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument(
        "--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument(
        "--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument(
        "--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument(
        "--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument(
        "--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")

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
        "--network", default='clip4cir_maple', type=str
    )
    parser.add_argument(
        "--asynchronous", default=False, action='store_true'
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
        "--bsc-loss", default='False', action='store_true'
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
        "--img2txt-model-path", type=str
    )
    parser.add_argument(
        "--txt2img", default=False, action='store_true'
    )
    parser.add_argument(
        "--clip-model-path", default='', type=str
    )
    parser.add_argument(
        "--clip-image-encoder-path", default='', type=str
    )


    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "encoder": args.encoder,
        "save_best": args.save_best,
        "asynchronous": args.asynchronous,
        "network": args.network,
        "maple_ctx_init": args.maple_ctx_init,
        "maple_n_ctx": args.maple_n_ctx,
        "maple_prompt_depth": args.maple_prompt_depth,
        "aligner": args.aligner,
        "corss_attn_layer": args.cross_attn_layer,
        "cross_attn_head": args.cross_attn_head,
        "bsc_loss": args.bsc_loss,
        "final": args.final,
        "num_workers": args.num_workers,
        "sum_combiner": args.sum_combiner,
        "fixed_image_encoder": args.fixed_image_encoder,
        "img2txt_model_path": args.img2txt_model_path,
        "txt2img": args.txt2img
    }


    if args.dataset.lower() == 'cirr':
        clip_finetune_cirr(args, **training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        clip_finetune_fiq(args, **training_hyper_params)
        # training_hyper_params.update(
        #     {'train_dress_types': ['toptee'], 'val_dress_types': ['toptee']})
        # clip_finetune_fiq(**training_hyper_params)
