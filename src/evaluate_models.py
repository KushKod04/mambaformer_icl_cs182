### our own EVAL script to evaluate the transformer & mamba models on new SinusoidalRegression and LongTermDependency data

# imports
from eval import get_model_from_run, eval_model, build_evals, compute_evals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import time
import torch


MODEL_PATHS = {
    'long_term_dependency': {
        'transformer': '../models/long_term_dependency/LTD_gpt2_embd512_layer8_lr1e-4_2025-04-22-00:55:01',
        'mamba': '../models/long_term_dependency/LTD_mamba_embd512_layer8_lr1e-4_2025-04-22-00:50:13',
    },
    'sinusoidal_regression': {
        'transformer': '../models/sinusoidal_regression/SR_tf_embd512_layer8_lr1e-4_2025-04-21-22:39:51',
        'mamba': '../models/sinusoidal_regression/SR_mamba_embd512_layer8_lr1e-4_2025-04-21-22:35:19',
    },
    'modulo_classification': {
        # 'transformer': '../models/modulo_classification/MC_gpt2_embd512_layer8_lr1e-4_dims-100-500_2025-05-04-20:13:23',  # 100 <= dims <= 500
        # 'mamba': '../models/modulo_classification/MC_mamba_embd512_layer8_lr1e-4_dims-100-500_2025-05-04-20:11:13',  # 100 <= dims <= 500
        'transformer': '../models/modulo_classification/MC_gpt2_embd512_layer8_lr1e-4_dims-20-80_2025-05-04-22:13:22',  # 20 <= dims <= 80
        'mamba': '../models/modulo_classification/MC_mamba_embd512_layer8_lr1e-4_dims-20-80_2025-05-04-22:11:23',  # 20 <= dims <= 80
    },
    'euclidean_distance': {
        'transformer': '../models/euclidean_distance/ED_tf_embd128_layer2_lr1e-4_2025-05-04-19:27:40',
        'mamba': '../models/euclidean_distance/ED_mamba_embd128_layer4_lr1e-4_2025-05-04-19:36:44',
    },
    'l1_distance': {
        'transformer': '../models/l1_distance/L1_tf_embd128_layer2_lr1e-4_2025-05-04-19:44:25',
        'mamba': '../models/l1_distance/L1_mamba_embd128_layer4_lr1e-4_2025-05-04-19:49:07',
    },
    'high_frequency': {
        'transformer': '../models/high_frequency/HF_mamba_embd128_layer4_lr1e-4_2025-05-04-23:05:09',
        'mamba': '../models/high_frequency/HF_tf_embd128_layer2_lr1e-4_2025-05-04-22:37:43',
    },
    'vector_manipulation': {
        'transformer': '../models/vector_manipulation/VM_gpt2_embd512_layer8_lr1e-4_2025-05-04-23:37:47',
        'mamba': '../models/vector_manipulation/VM_mamba_embd512_layer8_lr1e-4_2025-05-04-23:05:42',
    },
}


# def load_models():
#     ltd_gpt_model, ltd_gpt_conf = get_model_from_run(MODEL_PATHS['long_term_dependency']['transformer'], step=100000)
#     ltd_mamba_model, ltd_mamba_conf = get_model_from_run(MODEL_PATHS['long_term_dependency']['mamba'], step=100000)
#     sr_gpt_model, sr_gpt_conf = get_model_from_run(MODEL_PATHS['sinusoidal_regression']['transformer'], step=100000)
#     sr_mamba_model, sr_mamba_conf = get_model_from_run(MODEL_PATHS['sinusoidal_regression']['mamba'], step=100000)

#     ltd_gpt_model = ltd_gpt_model.cuda().eval()
#     ltd_mamba_model = ltd_mamba_model.cuda().eval()
#     sr_gpt_model = sr_gpt_model.cuda().eval()
#     sr_mamba_model = sr_mamba_model.cuda().eval()

#     return {
#         'ltd_gpt': (ltd_gpt_model, ltd_gpt_conf),
#         'ltd_mamba': (ltd_mamba_model, ltd_mamba_conf),
#         'sr_gpt': (sr_gpt_model, sr_gpt_conf),
#         'sr_mamba': (sr_mamba_model, sr_mamba_conf),
#     }


# shape is either (2*n_points+1,) or (points.end,)
# metrics = {
#     'mean': ...,
#     'std': ...,
#     'bootstrap_low': ...,
#     'bootstrap_high': ...,
# }

# for key, value in metrics.items():
#     print(f'key: {key}')
#     print(f'len(value) = {len(value)}')

# evaluation_kwargs = build_evals(conf)
# metrics = compute_evals(conf, [model], evaluation_kwargs)



def eval_increasing_n_points(task_name: str, min_points: int=11, max_points: int=41, step: int=2):
    """
        Evaluate the 2 models on increasing number of points.
        Args:
            task_name: which base task we are evaluating on
            min_points: minimum number of points to start evaluation on
            max_points: maximum number of points to end evaluation on
            step: step size for increasing number of points
    """
    assert task_name in MODEL_PATHS, f"Task '{task_name}' not found in MODEL_PATHS."

    save_dir = "eval_results"
    os.makedirs(save_dir, exist_ok=True)

    n_points_range = list(range(min_points, max_points+1, step))
    model_results = {model: [] for model in MODEL_PATHS[task_name]}

    print(f"\n=== Task: {task_name} ===")
    for model_type, path in MODEL_PATHS[task_name].items():
        print(f"\n→ Evaluating {model_type.upper()} model from: {path}")
        model, conf = get_model_from_run(path, step=100000)
        model = model.cuda().eval()

        for n_pts in n_points_range:
            metrics = eval_model(
                conf=conf,
                model=model,
                task_name=conf.training.task,
                data_name=conf.training.data,
                n_dims=conf.model.n_dims,
                n_points=n_pts,
                prompting_strategy="standard",
                num_eval_examples=1280,
                batch_size=64,
                data_sampler_kwargs=conf.training.data_sampler_kwargs,
                task_sampler_kwargs=conf.training.task_kwargs,
            )

            mean_error = np.mean(metrics["mean"])
            model_results[model_type].append((n_pts, mean_error))
            print(f"n_points = {n_pts} → Mean Error = {mean_error:.4f}")

    plt.figure(figsize=(10, 5))
    for model_type, data in model_results.items():
        n_pts_list, mean_errors = zip(*data)
        plt.plot(n_pts_list, mean_errors, label=model_type.upper(), marker='o')

    plt.axvline(x=11, color='red', linestyle='--', label='Train Range Start')
    plt.axvline(x=41, color='red', linestyle='--', label='Train Range End')
    plt.title(f"Evaluating {task_name}: Varying n_points", fontsize=18)
    plt.xlabel("Number of Points (n_points)", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{task_name}_n_points_eval.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


def eval_increasing_num_eval_examples(task_name: str, min_examples: int=64, max_examples: int=1920, step: int=64):
    """
        Evaluate the 2 models on increasing num_eval_examples.
        Args:
            task_name: which base task we are evaluating on
            min_examples: minimum number of eval examples to start evaluation on
            max_examples: maximum number of eval examples to end evaluation on
            step: step size for increasing number of eval examples
    """
    assert task_name in MODEL_PATHS, f"Task '{task_name}' not found in MODEL_PATHS."

    save_dir = "eval_results"
    os.makedirs(save_dir, exist_ok=True)

    num_eval_examples_range = list(range(min_examples, max_examples+1, step))
    model_results = {model: [] for model in MODEL_PATHS[task_name]}

    for model_type, path in MODEL_PATHS[task_name].items():
        print(f"\n→ Evaluating {model_type.upper()} model from: {path}")
        model, conf = get_model_from_run(path, step=100000)
        model = model.cuda().eval()

        for num_eval_examples in num_eval_examples_range:
            metrics = eval_model(
                conf=conf,
                model=model,
                task_name=conf.training.task,
                data_name=conf.training.data,
                n_dims=conf.model.n_dims,
                n_points=conf.training.curriculum.points.end,
                prompting_strategy="standard",
                num_eval_examples=num_eval_examples,
                batch_size=64,
                data_sampler_kwargs=conf.training.data_sampler_kwargs,
                task_sampler_kwargs=conf.training.task_kwargs,
            )

            mean_error = np.mean(metrics["mean"])
            model_results[model_type].append((num_eval_examples, mean_error))
            print(f"num_eval_examples = {num_eval_examples} → Mean Error = {mean_error:.4f}")

    plt.figure(figsize=(8, 6))
    for model_type, data in model_results.items():
        dims_list, mean_errors = zip(*data)
        plt.plot(dims_list, mean_errors, label=model_type.upper(), marker='o')

    plt.title(f"Evaluating {task_name}: num_eval_examples", fontsize=18)
    plt.xlabel("Number of Eval Examples", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{task_name}_num_examples_eval.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


# this code DOES NOT work (matrix multiplication dimension mismatch)
def eval_increasing_n_dims(task_name: str, min_dims: int=1, max_dims: int=20, step: int=1):
    """
        Evaluate the 2 models on increasing n_dims.
        Args:
            task_name: which base task we are evaluating on
            min_dims: minimum number of dimensions to start evaluation on
            max_dims: maximum number of dimensions to end evaluation on
            step: step size for increasing number of dims
    """
    assert task_name in MODEL_PATHS, f"Task '{task_name}' not found in MODEL_PATHS."

    save_dir = "eval_results"
    os.makedirs(save_dir, exist_ok=True)

    n_dims_range = list(range(min_dims, max_dims+1, step))
    model_results = {model: [] for model in MODEL_PATHS[task_name]}

    for model_type, path in MODEL_PATHS[task_name].items():
        print(f"\n→ Evaluating {model_type.upper()} model from: {path}")
        model, conf = get_model_from_run(path, step=100000)
        model = model.cuda().eval()

        for n_dims in n_dims_range:
            metrics = eval_model(
                conf=conf,
                model=model,
                task_name=conf.training.task,
                data_name=conf.training.data,
                n_dims=n_dims,
                n_points=conf.training.curriculum.points.end,
                prompting_strategy="standard",
                num_eval_examples=1280,
                batch_size=64,
                data_sampler_kwargs=conf.training.data_sampler_kwargs,
                task_sampler_kwargs=conf.training.task_kwargs,
            )

            mean_error = np.mean(metrics["mean"])
            model_results[model_type].append((n_dims, mean_error))
            print(f"n_dims = {n_dims} → Mean Error = {mean_error:.4f}")

    plt.figure(figsize=(8, 5))
    for model_type, data in model_results.items():
        dims_list, mean_errors = zip(*data)
        plt.plot(dims_list, mean_errors, label=model_type.upper(), marker='o')

    plt.title(f"Evaluation on Task: {task_name} (varying n_dims)")
    plt.xlabel("Input Dimension (n_dims)")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{task_name}_n_dims_eval.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


# this code is IRRELEVANT
def loss_spread(task_name: str):
    save_dir = 'eval_results'
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    print(f"\n=== Task: {task_name} ===")
    for model_type, path in MODEL_PATHS[task_name].items():
        print(f"\n→ Evaluating {model_type.upper()} model from: {path}")
        model, conf = get_model_from_run(path, step=100000)
        model = model.cuda().eval()

        metrics = eval_model(
            conf=conf,
            model=model,
            task_name=conf.training.task,
            data_name=conf.training.data,
            n_dims=conf.model.n_dims,
            n_points=conf.training.curriculum.points.end,
            prompting_strategy="standard",
            num_eval_examples=1280,
            batch_size=64,
            data_sampler_kwargs=conf.training.data_sampler_kwargs,
            task_sampler_kwargs=conf.training.task_kwargs,
        )
        results[model_type] = np.array(metrics["mean"])
    
    plt.figure(figsize=(8, 6))
    for model_type, losses in results.items():
        plt.hist(losses, bins=50, alpha=0.6, label=f"{model_type.upper()}")

    plt.xlabel("Per-Point Error", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(f"Loss Distribution for {task_name.replace('_', ' ').title()}", fontsize=18)
    plt.legend(fontsize=14)
    plot_path = os.path.join(save_dir, f"{task_name}_loss_distribution.png")
    plt.savefig(plot_path)
    plt.close()


def plot_loss_distribution(task_name: str, log_kde: bool = False):
    save_dir = 'eval_results'
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    print(f"\n=== Task: {task_name} ===")
    for model_type, path in MODEL_PATHS[task_name].items():
        print(f"\n→ Evaluating {model_type.upper()} model from: {path}")
        model, conf = get_model_from_run(path, step=100000)
        model = model.cuda().eval()

        metrics = eval_model(
            conf=conf,
            model=model,
            task_name=conf.training.task,
            data_name=conf.training.data,
            n_dims=conf.model.n_dims,
            n_points=conf.training.curriculum.points.end,
            prompting_strategy="standard",
            num_eval_examples=1280,
            batch_size=64,
            data_sampler_kwargs=conf.training.data_sampler_kwargs,
            task_sampler_kwargs=conf.training.task_kwargs,
        )
        flat_losses = np.array(metrics["mean"]).flatten()
        flat_losses = flat_losses[flat_losses > 1e-8]  # avoid log(0)
        results[model_type] = flat_losses
        if not log_kde:
            clipped = np.clip(metrics["mean"], 0, np.percentile(metrics["mean"], 99))  # uncomment for smoothed loss (clipped)
            results[model_type] = clipped

    plt.figure(figsize=(8, 6))
    for model_type, losses in results.items():
        sns.kdeplot(losses, label=model_type.upper(), fill=True, alpha=0.5)

    # smoothed loss distribution
    if not log_kde:
        plt.title(f"Smoothed Loss Distribution for {task_name.replace('_', ' ').title()}", fontsize=18)
        plt.xlabel("Per-Point Error (clipped @ 99th percentile)", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{task_name}_loss_kde.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

    # log-scale loss distribution
    else:
        plt.title(f"Log-Scale Loss Distribution for {task_name.replace('_', ' ').title()}", fontsize=18)
        plt.xlabel("Per-Point Error (log scale)", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.xscale("log")
        plt.legend(fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{task_name}_log_kde_loss.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")


def compare_prompting_strategies(task_name: str):

    save_path = f"eval_results/{task_name}_prompting_comparison.png"
    os.makedirs("eval_results", exist_ok=True)

    # model_errors = {model_type: [] for model_type in MODEL_PATHS[task_name]}
    model_errors = {model_type: {} for model_type in MODEL_PATHS[task_name]}
    prompting_strategies = ['standard', 'by_position', 'noisy_query', 'random_quadrants', 'fixed_dummy', 'wo_dummy']

    for model_type, path in MODEL_PATHS[task_name].items():
        print(f"Evaluating {model_type.upper()} model from: {path}")
        model, conf = get_model_from_run(path, step=100000)
        model = model.cuda().eval()

        for strategy in prompting_strategies:
            print(f"→ Strategy: {strategy}")
            strategy_success = True

            try:
                start = time.time()
                metrics = eval_model(
                    conf=conf,
                    model=model,
                    task_name=conf.training.task,
                    data_name=conf.training.data,
                    n_dims=conf.model.n_dims,
                    n_points=conf.training.curriculum.points.end,
                    prompting_strategy=strategy,
                    num_eval_examples=1280,
                    batch_size=64,
                    data_sampler_kwargs=conf.training.data_sampler_kwargs,
                    task_sampler_kwargs=conf.training.task_kwargs,
                )
                end = time.time()
                duration = end - start
                mean_error = np.mean(metrics["mean"])
                print(f"   Mean error = {mean_error:.4f} | Time = {duration:.2f} sec")
            except Exception as e:
                print(f"   Failed to evaluate strategy '{strategy}' due to error: {e}")
                mean_error = np.nan
            # model_errors[model_type].append(mean_error)
            model_errors[model_type][strategy] = mean_error

    successful_strategies = [strategy for strategy in prompting_strategies if all(model_errors[model_type].get(strategy, np.nan) is not np.nan for model_type in model_errors.keys())]
    x = np.arange(len(successful_strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.bar(x - width / 2, model_errors["transformer"], width, label="TRANSFORMER")
    # ax.bar(x + width / 2, model_errors["mamba"], width, label="MAMBA")
    for i, model_type in enumerate(model_errors.keys()):
        errors = [model_errors[model_type][s] for s in successful_strategies]
        ax.bar(x + (i - 0.5) * width, errors, width, label=model_type.upper())

    ax.set_ylabel("Mean Error")
    ax.set_xlabel("Prompting Strategy")
    ax.set_title(f"Prompting Strategy Comparison on {task_name.replace('_', ' ').title()}")
    ax.set_xticks(x)
    ax.set_xticklabels(successful_strategies, rotation=15)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot to: {save_path}")
    return save_path


def plot_per_point_error_comparison(task_name: str):
    save_dir = 'eval_results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task_name}_per_point_error.png")

    model_errors = {}

    for model_type, path in MODEL_PATHS[task_name].items():
        print(f"\nEvaluating {model_type.upper()} model from: {path}")
        model, conf = get_model_from_run(path, step=100000)
        model = model.cuda().eval()

        metrics = eval_model(
            conf=conf,
            model=model,
            task_name=conf.training.task,
            data_name=conf.training.data,
            n_dims=conf.model.n_dims,
            n_points=conf.training.curriculum.points.end,
            prompting_strategy="standard",
            num_eval_examples=1280,
            batch_size=64,
            data_sampler_kwargs=conf.training.data_sampler_kwargs,
            task_sampler_kwargs=conf.training.task_kwargs,
        )

        mean_error = np.array(metrics["mean"])
        model_errors[model_type] = mean_error

    plt.figure(figsize=(8, 6))
    x = np.arange(len(mean_error))
    for model_type, errors in model_errors.items():
        plt.plot(x, errors, label=model_type.upper(), linewidth=1.5)

    plt.title(f"Per-Point Error on {task_name.replace('_', ' ').title()}", fontsize=18)
    plt.xlabel("Point Index", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved per-point error plot to: {save_path}")


def plot_training_loss_from_tensorboard(task_name: str, tag: str="overall_loss/train"):
    """
    Plots the training loss curve from TensorBoard logs.

    Parameters:
        task_name (str): Type of task being evaluated.
        tag (str): The tag name used during training for logging the loss (default assumes "overall_loss/train").

    Returns:
        None
    """
    save_dir = 'eval_results'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))

    for model_type, model_path in MODEL_PATHS[task_name].items():
        tb_dir = os.path.join(model_path, "tensorboard")

        ea = event_accumulator.EventAccumulator(tb_dir)
        ea.Reload()

        available_tags = ea.Tags().get('scalars', [])
        if tag not in available_tags:
            print(f"Tag '{tag}' not found for {model_type} in {tb_dir}. Available tags are {available_tags}")
            continue

        events = ea.Scalars(tag)

        # Downsample to every `step_stride` (e.g., 1000)
        filtered_events = [event for event in events if event.step % 1000 == 0 and event.value < 5]

        steps = [event.step for event in filtered_events]
        values = [event.value for event in filtered_events]
        plt.plot(steps, values, label=model_type.upper())
    
    plt.title(f"Training Loss for {task_name.replace('_', ' ').title()}", fontsize=18)
    plt.xlabel("Training Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{task_name}_training_loss.png")
    plt.savefig(plot_path)
    print(f"Saved training loss plot to: {plot_path}")
    plt.close()


def plot_training_loss_from_wandb(task_name: str, tag: str="overall_loss/train"):
    """
    Plots the training loss curve from Weights & Biases logs.

    Parameters:
        task_name (str): Type of task being evaluated.
        tag (str): The tag name used during training for logging the loss (default assumes "overall_loss/train").
    """
    save_dir = 'eval_results'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))

    for model_type, model_path in MODEL_PATHS[task_name].items():
        log_dir = os.path.join(model_path, "wandb")
        losses, steps = [], []

        for file in os.listdir(log_dir):
            if file.startswith("events.out.tfevents"):
                ea = event_accumulator.EventAccumulator(os.path.join(log_dir, file))
                ea.Reload()

                available_tags = ea.Tags().get('scalars', [])
                if tag not in available_tags:
                    print(f"Tag '{tag}' not found for {model_type} in {log_dir}. Available tags are {available_tags}")
                    continue

                if tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    for event in events:
                        steps.append(event.step)
                        losses.append(event.value)

        plt.plot(steps, losses, label=model_type.upper())
    
    plt.title(f"Training Loss for {task_name.replace('_', ' ').title()}", fontsize=18)
    plt.xlabel("Training Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.ylim(0, max(losses) * 1.05)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{task_name}_training_loss.png")
    plt.savefig(plot_path)
    print(f"Saved training loss plot to: {plot_path}")
    plt.close()


def plot_training_loss_from_metrics(task_name: str, step: int=1, log: bool=False):
    """
    Plots the training loss curve from training_metrics.csv

    Parameters:
        task_name (str): Type of task being evaluated.
    """
    save_dir = 'eval_results'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))

    y_upper = y_min = 0

    for model_type, model_path in MODEL_PATHS[task_name].items():
        metrics_dir = os.path.join(model_path, "training_metrics.csv")
        df = pd.read_csv(metrics_dir)
        steps, loss = df["step"][::step], df["overall_loss/train"][::step]
        if log:
            loss = np.log(loss + 1e-8)

        plt.plot(steps, loss, label=model_type.upper())
        y_upper = max(y_upper, max(loss))
        y_min = min(y_min, min(loss))
    
    plt.title(f"Training {'Log ' if log else ''}Loss for {task_name.replace('_', ' ').title()}", fontsize=18)
    plt.ylabel("Log Loss" if log else "Loss", fontsize=14)
    # plt.title(f"Training Loss for {task_name.replace('_', ' ').title()}", fontsize=18)
    # plt.ylabel("Loss", fontsize=14)
    plt.ylim(y_min, y_upper * 1.05)
    plt.xlabel("Training Step", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()

    suffix = "_log" if log else ""
    plot_path = os.path.join(save_dir, f"{task_name}_training_loss_metrics{suffix}.png")
    plt.savefig(plot_path)
    print(f"Saved training loss plot to: {plot_path}")
    plt.close()


if __name__ == '__main__':
    # TODO: RUN THIS ONCE FOR EVERY MODEL
    # eval_increasing_n_points('long_term_dependency', min_points=5, max_points=65, step=2)
    # eval_increasing_n_points('sinusoidal_regression', min_points=5, max_points=65, step=2)
    # eval_increasing_n_points('modulo_classification', min_points=5, max_points=65, step=2)
    # eval_increasing_n_points('euclidean_distance', min_points=5, max_points=65, step=2)
    # eval_increasing_n_points('l1_distance', min_points=5, max_points=65, step=2)
    eval_increasing_n_points('vector_manipulation', min_points=5, max_points=65, step=2)
    # eval_increasing_n_points('high_frequency', min_points=5, max_points=65, step=2)

    # TODO: RUN THIS ONCE FOR EVERY MODEL
    # eval_increasing_num_eval_examples('long_term_dependency', min_examples=64, max_examples=2048, step=64)
    # eval_increasing_num_eval_examples('sinusoidal_regression', min_examples=64, max_examples=2048, step=64)
    # eval_increasing_num_eval_examples('modulo_classification', min_examples=64, max_examples=2048, step=64)
    # eval_increasing_num_eval_examples('euclidean_distance', min_examples=64, max_examples=2048, step=64)
    # eval_increasing_num_eval_examples('l1_distance', min_examples=64, max_examples=2048, step=64)
    eval_increasing_num_eval_examples('vector_manipulation', min_examples=64, max_examples=2048, step=64)
    # eval_increasing_num_eval_examples('high_frequency', min_examples=64, max_examples=2048, step=64)

    # TODO: RUN THIS ONCE FOR EVERY MODEL
    # plot_loss_distribution('long_term_dependency')
    # plot_loss_distribution('sinusoidal_regression')
    # plot_loss_distribution('modulo_classification')
    # plot_loss_distribution('modulo_classification', log_kde=True)
    # plot_loss_distribution('euclidean_distance')
    # plot_loss_distribution('euclidean_distance', log_kde=True)  # get the full log-loss KDE plot
    # plot_loss_distribution('l1_distance')
    # plot_loss_distribution('l1_distance', log_kde=True)  # get the full log-loss KDE plot
    plot_loss_distribution('vector_manipulation')
    plot_loss_distribution('vector_manipulation', log_kde=True)  # get the full log-loss KDE plot
    # plot_loss_distribution('high_frequency')
    # plot_loss_distribution('high_frequency', log_kde=True)  # get the full log-loss KDE plot

    # TODO: RUN THIS ONCE FOR EVERY MODEL
    # plot_training_loss_from_tensorboard('long_term_dependency')
    # plot_training_loss_from_tensorboard('sinusoidal_regression')
    # plot_training_loss_from_tensorboard('modulo_classification')
    # plot_training_loss_from_tensorboard('euclidean_distance')
    # plot_training_loss_from_tensorboard('l1_distance')
    # plot_training_loss_from_tensorboard('vector_manipulation')
    # plot_training_loss_from_tensorboard('high_frequency')

    # TODO: RUN THIS ONCE FOR EVERY MODEL
    # plot_per_point_error_comparison('long_term_dependency')
    # plot_per_point_error_comparison('sinusoidal_regression')
    # plot_per_point_error_comparison('modulo_classification')
    # plot_per_point_error_comparison('euclidean_distance')
    # plot_per_point_error_comparison('l1_distance')
    plot_per_point_error_comparison('vector_manipulation')
    # plot_per_point_error_comparison('high_frequency')

    # TODO: if tensorboard doesn't work, plot the training loss from metrics
    # plot_training_loss_from_metrics('modulo_classification')
    plot_training_loss_from_metrics('vector_manipulation', log=True)
    # plot_training_loss_from_metrics('high_frequency')

    pass

    # ############## IGNORE EVERYTHING BELOW THIS LINE ##############
    # plot_training_loss_from_wandb('modulo_classification')  # only run after model_100000.pt files exist

    # loss_spread('long_term_dependency')  # useless
    # loss_spread('sinusoidal_regression')  # useless
    # loss_spread('modulo_classification')  # useless
    # compare_prompting_strategies('long_term_dependency')  #useless
    # compare_prompting_strategies('sinusoidal_regression')  #useless

