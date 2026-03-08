import argparse
import numpy as np
import torch

import wandb
from typing import Literal

from data import load_attention, load_labels
from spectral_graph_features import SpectralConfig
from spectral_graph_features import spectral_features
from probes import run_probes


def main(config, run):
    sorted_image_names, y, binarizer = load_labels(config.annotations_path)
    if config.debug:
        # get just 2 images labels with the same shape
        sorted_image_names = sorted_image_names[:2]
        y = y[:2]

    print("Label distribution:")
    for i, label_name in enumerate(binarizer.classes_):
        print(f"{label_name}: {np.sum(y[:, i])} positive samples ({np.mean(y[:, i]) * 100:.2f}%)")
    print("-" * 30)

    if config.calculate_features:
        features_path = f"features_{run.name}.npz"
        wandb.config.update({'features_path': features_path}, allow_val_change=True)

        all_attention_weights = load_attention(
            config.attention_data_path,
            sorted_image_names
        )
        if config.debug:
            # get just 1 attention maps for only 2 images with the same shape
            all_attention_weights = all_attention_weights[:2]
            all_attention_weights = [x['down_blocks.1.attentions.0.transformer_blocks.0.attn2']
                                     for x in all_attention_weights]

        spectral_config = SpectralConfig(
            normalized_laplacian=config.normalized_laplacian,
            sparse_factor=config.sparse_factor,
            k_eigenvalues=config.k_eigenvalues,
            aggregation_method=config.aggregation_method,
            approximate_betweenness=config.approximate_betweenness,
            betweenness_sample_size=config.betweenness_sample_size,
            approximate_closeness=config.approximate_closeness,
            closeness_cutoff=config.closeness_cutoff,
        )
        # returns a dict of 1d arrays
        feature_dict = spectral_features(all_attention_weights, config=spectral_config)
        np.savez(features_path, **feature_dict)
    else:
        feature_dict = np.load(config.features_path)

    # --- Define Feature Sets for Probing ---
    all_feature_keys = sorted(feature_dict.keys())

    feature_sets = {
        "all_features": all_feature_keys,
        "eigenvalue_stats": [k for k in all_feature_keys if "eigenvalue" in k and "eigenvalue_" not in k],
        "all_eigenvalues": [k for k in all_feature_keys if "eigenvalue" in k],
        "fiedler_vector": [k for k in all_feature_keys if "fiedler_vector" in k],
        "eigenvector_centrality": [k for k in all_feature_keys if "eigenvector_centrality" in k],
        "spectral_entropy": [k for k in all_feature_keys if "spectral_entropy" in k],
        "betweenness_centrality": [k for k in all_feature_keys if "betweenness" in k],
        "degree_centrality": [k for k in all_feature_keys if "degree_centrality" in k],
        "closeness_centrality": [k for k in all_feature_keys if "closeness" in k],
        "freeman_centrality": [k for k in all_feature_keys if "freeman_centrality" in k],
    }
    
    # --- Cross-validation per label per feature set ---
    for i, label_name in enumerate(binarizer.classes_):
        y_label = y[:, i]

        for feature_set_name, feature_keys in feature_sets.items():
            
            # Filter out keys that are not in the loaded features
            existing_feature_keys = [k for k in feature_keys if k in feature_dict]
            if not existing_feature_keys:
                print(f"Skipping feature set '{feature_set_name}' as no features were found in the data.")
                continue
            
            # Stack features for the current set
            x_set = np.stack([feature_dict[k] for k in existing_feature_keys], axis=1)
            
            # Handle potential NaNs or Infs
            if np.isnan(x_set).any() or np.isinf(x_set).any():
                print(f"Warning: NaNs or Infs found in feature set '{feature_set_name}'. Replacing with 0.")
                x_set = np.nan_to_num(x_set, nan=0.0, posinf=0.0, neginf=0.0)

            run_probes(x_set, y_label, config, feature_set_name, label_name)


if __name__ == "__main__":
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (disables wandb).')
    parser.add_argument('--recalculate', action='store_true', help='Recalculate features.')
    parser.add_argument('--features_path', type=str, default='features.npy', help='Path to the features file.')
    parser.add_argument('--annotations_path', type=str, default='./annotations.xml', help='Path to the annotations file.')
    parser.add_argument('--attention_data_path', type=str, default='/Users/arielkeslassy/Documents/reichman/courses/SNA/experiments/data', help='Path to the attention data directory.')
    args = parser.parse_args()

    default_config = {
        'debug': args.debug,
        'calculate_features': args.recalculate,
        'features_path': args.features_path,
        'annotations_path': args.annotations_path,
        'attention_data_path': args.attention_data_path,
        'n_splits': 5,
        'rng': 42,
        'logistic_regression_max_iter': 2000,
        'inner_cv_splits': 4,
        'normalized_laplacian': True,
        'sparse_factor': 0.1,
        'k_eigenvalues': 10,
        'aggregation_method': 'all',
        'approximate_betweenness': True,  # 5-20x faster
        'betweenness_sample_size': 100,  # Good balance
        'approximate_closeness': True,  # 2-10x faster
        'closeness_cutoff': 5,  # Good balance
    }

    wandb_mode: Literal["online", "disabled"] = "disabled" if args.debug else "online"

    with wandb.init(project="spectral-graph-failure-analysis",
                    config=default_config,
                    mode=wandb_mode,
                    save_code=True) as run:
        run.log_code(root=".",
                     exclude_fn=lambda path: path.endswith("tests.py"),
                     include_fn=lambda path: 'spectral_graph_features.py' in path or
                                             'data.py' in path or
                                             'main.py' in path)
        main(run.config, run)
