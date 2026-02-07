#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Full Factorial Ablation Study (Refactored)
==========================================

Runs systematic ablation study using the Common BaseArchitecture.
- Architecture Params (Growth, Compression, Depth)
- Batch Size
- Resolution
"""

import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, List

# Add paths
sys.path.append(str(Path(__file__).parent.parent))        # classification/
sys.path.append(str(Path(__file__).parent.parent.parent)) # template/

from config import ResearchConfig
from common.experiments.base_ablation import BaseAblationRunner
from src.models import create_model, get_model_metrics
from src.data_loader import validate_dataset_directory, load_dataset, split_dataset
from src.training import (
    train_model, compile_model, setup_multi_gpu,
    benchmark_inference, get_device_info
)
from src.visualization import close_all_figures
from sklearn.metrics import f1_score


class ClassificationAblationRunner(BaseAblationRunner):
    """
    Handles Classification-specific ablation logic.
    Inherits core runner loop from Common Base.
    """
    
    def __init__(self, results_dir: Path, data: tuple, config: ResearchConfig, seeds: List[int]):
        super().__init__(results_dir, seeds)
        self.X, self.Y, self.num_classes = data
        self.config = config
        self.current_search_space = {}
        
        # GPU Strategy
        self.strategy = setup_multi_gpu()
        
    def set_search_space(self, space: Dict[str, List[Any]]):
        """Allow dynamic updating of search space for multi-phase runs."""
        self.current_search_space = space
        
    def get_search_space(self) -> Dict[str, List[Any]]:
        return self.current_search_space
        
    def run_experiment(self, exp_config: Dict[str, Any], seed: int) -> Dict[str, float]:
        """
        Execute single classification experiment.
        """
        # 1. reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # 2. Extract params (with defaults from main config if missing)
        gr = exp_config.get('growth_rate', self.config.model.growth_rate)
        comp = exp_config.get('compression', self.config.model.compression)
        depth = exp_config.get('depth', self.config.model.depth)
        batch = exp_config.get('batch_size', self.config.training.batch_size)
        res = exp_config.get('resolution', self.config.data.img_size[0])
        lr = exp_config.get('learning_rate', self.config.training.learning_rate)
        
        # 3. Data Handling (Resize if needed)
        # Optimization: Use pre-loaded self.X/Y if resolution matches, else reload/resize
        current_res = self.config.data.img_size[0]
        
        if res != current_res:
            # Resolution changed - we must reload/resize
            # In a real heavy scenario, we might resize in-memory, but here we reload for simplicity/safety
            # knowing that load_dataset handles resizing.
            X_exp, Y_exp = load_dataset(
                self.config.data.data_dir, 
                categories=[str(i) for i in range(self.num_classes)], # dummy categories if already validated
                img_size=(res, res),
                max_images_per_class=self.config.data.max_images_per_class,
                show_progress=False
            )
        else:
            X_exp, Y_exp = self.X, self.Y
            
        # 4. Split
        splits = split_dataset(X_exp, Y_exp, seed=seed)
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # 5. Model
        inp_shape = (res, res, 3)
        with self.strategy.scope():
            model = create_model(
                input_shape=inp_shape,
                num_classes=self.num_classes,
                growth_rate=gr,
                compression=comp,
                depth=depth,
                dropout_rate=self.config.model.dropout_rate,
                initial_filters=self.config.model.initial_filters
            )
            model = compile_model(model, learning_rate=lr)
            
        # 6. Train
        # Define run directory for TensorBoard
        run_name = f"gr{gr}_c{comp}_d{depth}_b{batch}_r{res}_s{seed}"
        run_dir = self.results_dir / "runs" / run_name
        
        history = train_model(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            run_dir=run_dir,
            epochs=self.config.training.epochs,
            batch_size=batch,
            early_stopping_patience=self.config.training.early_stopping_patience,
            verbose=0
        )
        
        # 7. Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        macro_f1 = f1_score(y_test, y_pred, average='macro') * 100
        
        # Metrics
        metrics_info = get_model_metrics(model)
        latency = benchmark_inference(model, inp_shape, warmup_runs=2, benchmark_runs=5)
        
        tf.keras.backend.clear_session()
        close_all_figures()
        
        return {
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'macro_f1': macro_f1,
            'val_accuracy': max(history.get('val_accuracy', [0])) * 100,
            'total_params': metrics_info.total_params,
            'inference_ms': latency['batch_1']['mean_ms'],
            # Log specific params for the record
            'growth_rate': gr,
            'compression': comp, 
            'depth': str(depth),
            'batch_size': batch,
            'resolution': res,
            'learning_rate': lr
        }


def run_ablation(quick_test: bool = False, single_seed: bool = False):
    print("=" * 70)
    print("CLASSIFICATION ABLATION STUDY (Refactored)")
    print("=" * 70)
    
    config = ResearchConfig()
    
    # Overrides for quick test
    if quick_test:
        config.training.epochs = 2
        config.data.img_size = (32, 32) # Smaller for speed
    
    # Seeds
    use_single = single_seed or not config.training.use_multiple_seeds
    seeds = [config.training.seeds[0]] if use_single else config.training.seeds
    
    # 1. Load Data ONLY ONCE
    print("\n[1/3] Loading Data...")
    try:
        categories, _ = validate_dataset_directory(config.data.data_dir, min_classes=2)
        X, Y = load_dataset(
            config.data.data_dir, categories, 
            img_size=config.data.img_size,
            max_images_per_class=config.data.max_images_per_class
        )
    except Exception as e:
        print(f"[!] Error loading data: {e}")
        return None
        
    num_classes = len(categories)
    
    # 2. Initialize Runner
    runner = ClassificationAblationRunner(
        results_dir=config.output.results_dir,
        data=(X, Y, num_classes),
        config=config,
        seeds=seeds
    )
    
    all_dfs = []
    
    # 3. Run Groups (Phase 1, 2, 3)
    
    # --- Group 1: Architecture ---
    print("\n💠 GROUP 1: ARCHITECTURE SEARCH")
    runner.set_search_space({
        'growth_rate': config.ablation.growth_rates,
        'compression': config.ablation.compressions,
        'depth': config.ablation.depths
    })
    df_arch = runner.run(quick_test)
    if not df_arch.empty:
        df_arch['group'] = 'architecture'
        all_dfs.append(df_arch)
        
    # --- Group 2: Batch Size ---
    print("\n💠 GROUP 2: BATCH SIZE SEARCH")
    runner.set_search_space({
        'batch_size': config.ablation.batch_sizes
        # Note: Keeps default arch params from config
    })
    df_batch = runner.run(quick_test)
    if not df_batch.empty:
        df_batch['group'] = 'batch_size'
        all_dfs.append(df_batch)
        
    # --- Group 3: Resolution ---
    print("\n💠 GROUP 3: RESOLUTION SEARCH")
    runner.set_search_space({
        'resolution': config.ablation.resolutions
        # Note: Keeps default arch params from config
    })
    df_res = runner.run(quick_test)
    if not df_res.empty:
        df_res['group'] = 'resolution'
        all_dfs.append(df_res)

    # --- Group 4: Learning Rate ---
    print("\n💠 GROUP 4: LEARNING RATE SEARCH")
    runner.set_search_space({
        'learning_rate': config.ablation.learning_rates
    })
    df_lr = runner.run(quick_test)
    if not df_lr.empty:
        df_lr['group'] = 'learning_rate'
        all_dfs.append(df_lr)
        
    # 4. Final Aggregation
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        final_path = config.output.results_dir / "ablation_full_combined.csv"
        full_df.to_csv(final_path, index=False)
        print(f"\n✅ All studies complete. Combined results: {final_path}")
        return full_df
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--single-seed', action='store_true')
    args = parser.parse_args()
    
    run_ablation(args.quick, args.single_seed)