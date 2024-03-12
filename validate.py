import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from utils.utils import extract_index_features, collate_fn, element_wise_sum, device
from modeling import train_models_map
import pdb


def compute_fiq_val_metrics(args, relative_val_dataset: FashionIQDataset, model, tar_indexfeats: torch.tensor,
                            index_names: List[str]) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_fiq_val_predictions(args, model, relative_val_dataset, index_names, tar_indexfeats)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    tar_indexfeats = F.normalize(tar_indexfeats, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ tar_indexfeats.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(args, model, relative_val_dataset: FashionIQDataset, index_names: List[str], tar_indexfeats: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, tar_indexfeats))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, model.clip.visual.output_dim)).to(device, non_blocking=True)
    target_names = []


    for reference_names, reference_image, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)
        reference_image = reference_image.to(device)
        # Compute the predicted features
        with torch.no_grad():
            # prompts_init = args.maple_ctx_init
            # init_len = len(prompts_init.split())
            # prompts_init_token = clip.tokenize(prompts_init).to(device)
            # prompts_init_token = prompts_init_token.expand(text_inputs.shape[0], -1)
            # tokenized_prompts = torch.cat((prompts_init_token[:, :1 + init_len], \
            #                                text_inputs[:, 1: 77 - init_len]), dim=-1)
            if args.fixed_image_encoder:
                fixed_imgfeats = model.fixed_image_encoder(reference_image)
            if args.final:
                if args.fixed_image_encoder:
                    if args.txt2img:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                        ref_deep_compound_prompts_vision = model.prompt_learner(text_inputs, fixed_imgfeats)
                    else:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                            ref_deep_compound_prompts_text = model.prompt_learner(text_inputs, fixed_imgfeats)
                else:
                    if args.txt2img:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                        ref_deep_compound_prompts_vision = model.prompt_learner(text_inputs)
                    else:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                            ref_deep_compound_prompts_text = model.prompt_learner(text_inputs)
            else:
                if args.fixed_image_encoder:
                    if args.txt2img:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                        ref_deep_compound_prompts_vision = model.prompt_learner(text_inputs, fixed_imgfeats)
                    else:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                            ref_deep_compound_prompts_text = model.prompt_learner(text_inputs, fixed_imgfeats)
                else:
                    if args.txt2img:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                        ref_deep_compound_prompts_vision = model.prompt_learner(text_inputs)
                    else:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                            ref_deep_compound_prompts_text = model.prompt_learner(text_inputs)
            text_feats = model.text_encoder(ref_prompts, ref_tokenized_prompts, ref_deep_compound_prompts_text)
            
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with a single tensor
            if text_feats.shape[0] == 1:
                ref_imgfeats = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                ref_imgfeats = torch.stack(itemgetter(*reference_names)(name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            
            if args.sum_combiner:
                batch_predfeats = F.normalize(ref_imgfeats + text_feats, dim=-1)
            else:
                batch_predfeats = model.combiner(ref_imgfeats, text_feats)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predfeats, dim=-1)))
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def fashioniq_val_retrieval(args, dress_type: str, model, preprocess: callable):
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features(args, classic_val_dataset, model)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics(args, relative_val_dataset, model, index_features, index_names)


def compute_cirr_val_metrics(args, relative_val_dataset: CIRRDataset, model, index_features: torch.tensor,
                             index_names: List[str]) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(args, model, relative_val_dataset, index_names, index_features)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    pdb.set_trace()
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(args, model, relative_val_dataset: CIRRDataset, 
                                  index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, model.clip.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_reference_image, batch_target_names, captions, batch_group_members in tqdm(relative_val_loader):  # Load data
        text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        pdb.set_trace()
        batch_group_members = np.array(batch_group_members).T.tolist()
        batch_reference_image = batch_reference_image.to(device)
        # Compute the predicted features
        with torch.no_grad():

            # prompts_init = args.maple_ctx_init
            # init_len = len(prompts_init.split())
            # prompts_init_token = clip.tokenize(prompts_init).to(device)
            # prompts_init_token = prompts_init_token.expand(text_inputs.shape[0], -1)
            # tokenized_prompts = torch.cat((prompts_init_token[:, :1 + init_len], \
                                        #    text_inputs[:, 1: 77 - init_len]), dim=-1)
            if args.fixed_image_encoder:
                batch_fixed_features = model.fixed_image_encoder(batch_reference_image)
            if args.final:
                if args.fixed_image_encoder:
                    ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                        ref_deep_compound_prompts_text = model.prompt_learner(text_inputs, batch_fixed_features)
                else:
                    ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                        ref_deep_compound_prompts_text = model.prompt_learner(text_inputs)
            else:
                if args.fixed_image_encoder:
                    ref_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                        ref_deep_compound_prompts_vision = model.prompt_learner(text_inputs, batch_fixed_features)
                else:
                    ref_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                        ref_deep_compound_prompts_vision = model.prompt_learner(text_inputs)
            text_feats = model.text_encoder(ref_prompts, ref_tokenized_prompts, ref_deep_compound_prompts_text)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_feats.shape[0] == 1:
                ref_imgfeats = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                ref_imgfeats = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            if args.sum_combiner:
                batch_predicted_features = F.normalize(ref_imgfeats + text_feats, dim=-1)
            else:
                batch_predicted_features = model.combiner(ref_imgfeats, text_feats)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


def cirr_val_retrieval(args, model, preprocess: callable):
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    index_features, index_names = extract_index_features(args, classic_val_dataset, model)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics(args, relative_val_dataset, model, index_features, index_names)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--sum-combiner", default=False, action='store_true')
    parser.add_argument("--combiner-path", type=Path, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    
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
        "--final", default=False, action='store_true'
    )
    parser.add_argument(
        "--model-s1-path", type=str
    )
    parser.add_argument(
        "--model-s2-path", type=str
    )
    parser.add_argument(
        "--network", type=str
    )
    parser.add_argument(
        "--asynchronous", default=False, action='store_true'
    )
    parser.add_argument(
        "--combiner", type=str
    )
    parser.add_argument(
        "--aligner", default=False, action='store_true'
    )
    parser.add_argument(
        "--num-workers", default=8, type=int
    )
    parser.add_argument(
        "--fixed-image-encoder", default=False, action='store_true'
    )
    parser.add_argument(
        "--txt2img", default=False, action='store_true'
    )
    parser.add_argument(
        "--embed-size", default=512, type=int
    )
    parser.add_argument(
        "--mu", default=0.1, type=float
    )
    parser.add_argument(
        "--router", default=False, action='store_true'
    )
    parser.add_argument(
        "--optimizer", default='combiner', type=str
    )
    parser.add_argument(
        "--bsc-loss", default=False, action='store_true'
    )
    parser.add_argument(
        "--cross-attn-layer", default=4, type=int
    )
    parser.add_argument(
        "--cross-attn-head", default=2, type=int
    )
    parser.add_argument(
        "--img2txt-model-path", default='', type=str
    )

    args = parser.parse_args()

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

    input_dim = model.clip.visual.input_resolution
    clip_preprocess = model.clip_preprocess

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        model.clip.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.model_s1_path:
        print("Trying to load the model")
        ckpt = torch.load(args.model_s1_path, map_location=device)
        model.load_state_dict(ckpt[list(ckpt.keys())[-1]])
        

    if args.model_s2_path:
        print("Trying to load combiner")
        ckpt = torch.load(args.model_s2_path, map_location=device)
        model.combiner.load_state_dict(ckpt[list(ckpt.keys())[-1]])

    model = model.to(device)
    model.eval().float()

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess


    if args.dataset.lower() == 'cirr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(args, model, preprocess)

        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")
        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"Avg. = {(recall_at5 + group_recall_at1) / 2}")
        

    elif args.dataset.lower() == 'fashioniq':
        average_recall10_list = []
        average_recall50_list = []

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval(args, 'shirt', model, preprocess)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval(args, 'dress', model, preprocess)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval(args, 'toptee', model, preprocess)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)

        print(f"\n{shirt_recallat10 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat50 = }")

        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")
        print(f"Avg recall = {(mean(average_recall10_list) + mean(average_recall50_list)) / 2}")
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    main()
