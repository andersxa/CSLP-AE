import sys
import wandb
from split_model import SplitLatentModel
import torch
import numpy as np
import torch.optim as optim
from utils import get_results, get_eval_results, get_split_latents, split_do_tsne, plot_latents, CustomLoader, fit_knn_fn, fit_etc_fn
from conversion_utils import get_full_conversion_results
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--model_save_dir', type=str, default='./')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--recon_type', type=str, default='mse')
parser.add_argument('--content_cosine', type=int, default=1)

parser.add_argument('--data_line', type=str, default='simple')

parser.add_argument('--final_div_factor', type=int, default=10)
parser.add_argument('--initial_lr', type=float, default=0.0001)
parser.add_argument('--max_lr', type=float, default=0.0001)
parser.add_argument('--pct_start', type=float, default=0.5)

parser.add_argument('--sub_cross_s_enabled', type=int, default=0)
parser.add_argument('--sub_cross_s_weight', type=float, default=1.0)
parser.add_argument('--task_cross_t_enabled', type=int, default=0)
parser.add_argument('--task_cross_t_weight', type=float, default=1.0)

parser.add_argument('--recon_enabled', type=int, default=0)
parser.add_argument('--recon_weight', type=float, default=1.0)

parser.add_argument('--scramble_permute_enabled', type=int, default=0)
parser.add_argument('--scramble_permute_weight', type=float, default=1.0)

parser.add_argument('--conversion_permute_enabled', type=int, default=0)
parser.add_argument('--conversion_permute_weight', type=float, default=1.0)

parser.add_argument('--quadruplet_permute_enabled', type=int, default=0)
parser.add_argument('--quadruplet_permute_F_enabled', type=int, default=0)
parser.add_argument('--quadruplet_permute_weight', type=float, default=1.0)

parser.add_argument('--sub_contra_s_enabled', type=int, default=0)
parser.add_argument('--sub_contra_s_weight', type=float, default=1.0)
parser.add_argument('--task_contra_t_enabled', type=int, default=0)
parser.add_argument('--task_contra_t_weight', type=float, default=1.0)

parser.add_argument('--latent_permute_s_enabled', type=int, default=0)
parser.add_argument('--latent_permute_s_weight', type=float, default=1.0)
parser.add_argument('--latent_permute_t_enabled', type=int, default=0)
parser.add_argument('--latent_permute_t_weight', type=float, default=1.0)

parser.add_argument('--restored_permute_s_enabled', type=int, default=0)
parser.add_argument('--restored_permute_s_weight', type=float, default=1.0)
parser.add_argument('--restored_permute_t_enabled', type=int, default=0)
parser.add_argument('--restored_permute_t_weight', type=float, default=1.0)

parser.add_argument('--sub_content_enabled', type=int, default=0)
parser.add_argument('--sub_content_weight', type=float, default=1.0)
parser.add_argument('--task_content_enabled', type=int, default=0)
parser.add_argument('--task_content_weight', type=float, default=1.0)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--use_tqdm', type=int, default=1)

parser.add_argument('--group', type=str, default='')

parser.add_argument('--override_seed', type=int, default=None)

parser.add_argument('--extra_tags', type=str, default='')

parser.add_argument('--full_eval', type=int, default=1)

parser.add_argument('--eval_every', type=int, default=70)
parser.add_argument('--sched_mode', type=str, default='max')
parser.add_argument('--sched_patience', type=int, default=5)
parser.add_argument('--sched_factor', type=float, default=0.5)
parser.add_argument('--old_sched', type=int, default=1)
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--add_name', type=str, default='')
parser.add_argument('--conversion_N', type=int, default=2000)
parser.add_argument('--extra_classifiers', type=int, default=1)
parser.add_argument('--conversion_results', type=int, default=1)

args, unknown = parser.parse_known_args()

loss_to_notation = {
    'recon': ['R'],
    'sub_contra_s': ['SL', 'CR:s'],
    'task_contra_t': ['SL', 'CR:t'],
    'latent_permute_s': ['SL', 'LP:s'],
    'latent_permute_t': ['SL', 'LP:t'],
    'restored_permute_s': ['SL', 'RP:s'],
    'restored_permute_t': ['SL', 'RP:t'],
    'sub_content': ['SL', 'C:s'],
    'task_content': ['SL', 'C:t'],
    'sub_cross_s': ['CE:s'],
    'task_cross_t': ['CE:t'],
    'scramble_permute': ['SP'],
    'conversion_permute': ['CP'],
    'quadruplet_permute': ['QP'],
    'quadruplet_permute_f': ['QPf'],
}

if __name__ == '__main__':
    print(args, file=sys.stdout, flush=True)
    if args.random_seed:
        SEED = np.random.randint(0, 2**32 - 1)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    elif args.override_seed is not None:
        SEED = args.override_seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    else:
        SEED = 3242342323
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    IN_CHANNELS = 30
    NUM_LAYERS = args.num_layers
    KERNEL_SIZE = 4
    
    USE_TQDM = args.use_tqdm
    
    OLD_SCHED = bool(args.old_sched)
    
    model = SplitLatentModel(IN_CHANNELS, args.channels, args.latent_dim, NUM_LAYERS, KERNEL_SIZE, recon_type=args.recon_type, content_cosine=args.content_cosine)
    with torch.no_grad():
        data_dict = torch.load(args.data_dir+f"{args.data_line}_data.pt")
        data_dict["data"] = (data_dict["data"] - data_dict["data_mean"]) / data_dict["data_std"]
        loader = CustomLoader(data_dict, split='train')
        test_loader = CustomLoader(data_dict, split='test')
        eval_loader = CustomLoader(data_dict, split='dev')
        del data_dict
    
    all_losses = ["recon", "sub_contra_s", "task_contra_t", "latent_permute_s", "latent_permute_t", "restored_permute_s", "restored_permute_t", "sub_content", "task_content", "sub_cross_s", "task_cross_t", "scramble_permute", "conversion_permute", "quadruplet_permute", "quadruplet_permute_F"]
    losses = []
    loss_weights = defaultdict(lambda: 1.0)
    if args.recon_enabled:
        losses.append("recon")
        loss_weights["recon"] = args.recon_weight
    if args.sub_contra_s_enabled:
        losses.append("sub_contra_s")
        loss_weights["sub_contra_s"] = args.sub_contra_s_weight
    if args.task_contra_t_enabled:
        losses.append("task_contra_t")
        loss_weights["task_contra_t"] = args.task_contra_t_weight
    if args.latent_permute_s_enabled:
        losses.append("latent_permute_s")
        loss_weights["latent_permute_s"] = args.latent_permute_s_weight
    if args.latent_permute_t_enabled:
        losses.append("latent_permute_t")
        loss_weights["latent_permute_t"] = args.latent_permute_t_weight
    if args.restored_permute_s_enabled:
        losses.append("restored_permute_s")
        loss_weights["restored_permute_s"] = args.restored_permute_s_weight
    if args.restored_permute_t_enabled:
        losses.append("restored_permute_t")
        loss_weights["restored_permute_t"] = args.restored_permute_t_weight
    if args.sub_content_enabled:
        losses.append("sub_content")
        loss_weights["sub_content"] = args.sub_content_weight
    if args.task_content_enabled:
        losses.append("task_content")
        loss_weights["task_content"] = args.task_content_weight
    if args.sub_cross_s_enabled:
        losses.append("sub_cross_s")
        loss_weights["sub_cross_s"] = args.sub_cross_s_weight
    if args.task_cross_t_enabled:
        losses.append("task_cross_t")
        loss_weights["task_cross_t"] = args.task_cross_t_weight
    if args.scramble_permute_enabled:
        losses.append("scramble_permute")
        loss_weights["scramble_permute"] = args.scramble_permute_weight
    if args.conversion_permute_enabled:
        losses.append("conversion_permute")
        loss_weights["conversion_permute"] = args.conversion_permute_weight
    if args.quadruplet_permute_enabled:
        if args.quadruplet_permute_F_enabled:
            losses.append("quadruplet_permute_F")
            loss_weights["quadruplet_permute_F"] = args.quadruplet_permute_weight
        else:
            losses.append("quadruplet_permute")
            loss_weights["quadruplet_permute"] = args.quadruplet_permute_weight
    
    model.set_losses(
        batch_size=args.batch_size,
        losses=losses,
        loader=loader,
        loss_weights=loss_weights,
    )
    
    
    numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {numel}", file=sys.stdout, flush=True)
    
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    with torch.no_grad():
        model.losses()
    EFFECTIVE_BATCH_SIZE = loader.total_samples
    print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}", file=sys.stdout, flush=True)
    
    loss_notation = [n for l in model.used_losses for n in loss_to_notation[l.lower()]]
    loss_notation = sorted(set(loss_notation), key=loss_notation.index)
    loss_tags = "_".join(loss_notation).replace(":", "").replace("CRs_CRt", "CR").replace("LPs_LPt", "LP").replace("RPs_RPt", "RP").replace("Cs_Ct", "C").replace("CEs_CEt", "CE")
    print("Loss tags:", loss_tags, file=sys.stdout, flush=True)
    print("Used losses:", model.used_losses, file=sys.stdout, flush=True)
    
    BATCHES = (args.epochs * loader.size) // EFFECTIVE_BATCH_SIZE
    
    div_factor = args.max_lr / args.initial_lr
    if OLD_SCHED:
        scheduler = optim.lr_scheduler.OneCycleLR(
           optimizer,
           div_factor=div_factor,
           max_lr=args.max_lr,
           steps_per_epoch=1,
           epochs=BATCHES,
           three_phase=False,
           pct_start=args.pct_start,
           final_div_factor=args.final_div_factor,
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=args.sched_mode,
            factor=args.sched_factor,
            patience=args.sched_patience,
        )
    
    #%%
    wandb_config = {
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_lr": args.max_lr,
        "initial_lr": args.initial_lr,
        "pct_start": args.pct_start,
        "final_div_factor": args.final_div_factor,
        "data_line": args.data_line,
        "batches": BATCHES,
        "losses": model.used_losses,
        "seed": SEED,
        "in_channels": IN_CHANNELS,
        "channels": args.channels,
        "latent_dim": args.latent_dim,
        "num_layers": NUM_LAYERS,
        "num_params": numel,
        "effective_latent_dim": model.effective_latent_dim,
        "latent_seqs": model.latent_seqs,
        "recon_type": model.recon_type,
        "loss_tags": loss_tags,
        "eval_every": args.eval_every,
        "conversion_N": args.conversion_N,
    }
    for l in model.used_losses:
        wandb_config[f"{l}_weight"] = model.loss_weights[l]
    for l in all_losses:
        wandb_config[f"{l}_enabled"] = l in model.used_losses
    extra_tags = []
    if len(args.extra_tags) > 0:
        extra_tags = args.extra_tags.split(",")
    group = None
    if len(args.group) > 0:
        group = args.group
    wandb.init(
        project="converting-erps",
        config=wandb_config,
        group=group,
        name=f'{loss_tags}-{args.add_name}-{np.random.randint(0, 1000):03d}' if len(args.add_name) > 0 else f'{loss_tags}-{np.random.randint(0, 1000):03d}',
        tags=["split-model", "simple", loss_tags] + model.used_losses + extra_tags,
    )
    
    wandb.run.log_code(include_fn=lambda path: path.endswith("train.py") or path.endswith("split_model.py") or path.endswith("utils.py"))
    
    #%%
    loss_list = defaultdict(list)
    with tqdm(range(BATCHES), unit_scale=EFFECTIVE_BATCH_SIZE, disable=not USE_TQDM, file=sys.stdout) as pbar:
        for i in pbar:
            model.train()
            optimizer.zero_grad()
            x, loss_dict = model.losses()
            total_loss = sum((model.loss_weights[v] * loss_dict[v] for v in model.used_losses))
            total_loss.backward()
            optimizer.step()
            if OLD_SCHED:
                scheduler.step()
            if i % args.eval_every == 0:
                with torch.no_grad():
                    subject_latents, task_latents, subjects, tasks, runs, losses = get_split_latents(model, eval_loader, eval_loader.get_dataloader(batch_size=model.batch_size, random_sample=False))
                    eval_results = get_eval_results(subject_latents, task_latents, subjects, tasks, split="eval")
                    eval_results.update({f'eval/{v}': losses[v] for v in list(model.used_losses)})
                    eval_results.update({f'train/{v}': loss_dict[v] for v in list(model.used_losses)})
                    eval_results.update({
                        'train/total_loss': total_loss,
                        'total_samples': loader.total_samples,
                        'train/sched_lr': optimizer.param_groups[0]['lr'],
                        'train/s_scale': model.subj_logit_scale.exp().item(),
                        'train/t_scale': model.task_logit_scale.exp().item(),
                    })
                    wandb.log(eval_results)
                    if not OLD_SCHED:
                        scheduler.step(eval_results["XGB/eval/task/score"])
    
    #%%
    model.loader = test_loader
    model.eval()
    print('Evaluating...', file=sys.stdout, flush=True)
    subject_latents, task_latents, subjects, tasks, runs, losses = get_split_latents(model, test_loader, test_loader.get_dataloader(batch_size=model.batch_size, random_sample=False))
    test_results = get_results(subject_latents, task_latents, subjects, tasks, split=test_loader.split, off_class_accuracy=args.full_eval)
    wandb.log(test_results)
    if args.conversion_results:
        mse_results = get_full_conversion_results(model, test_loader, subject_latents, task_latents, args.conversion_N)
        wandb.log(mse_results)
    if args.extra_classifiers:
        test_knn_results = get_results(subject_latents, task_latents, subjects, tasks, clf='KNN', fit_clf=fit_knn_fn, split=test_loader.split)
        wandb.log(test_knn_results)
        test_etc_results = get_results(subject_latents, task_latents, subjects, tasks, clf='ETC', fit_clf=fit_etc_fn, split=test_loader.split)
        wandb.log(test_etc_results)
    figure_results = {}
    #%%
    print('Reconstructing...', file=sys.stdout, flush=True)
    with torch.no_grad():
        x = model.get_x_hat({})
    #%%
    print('Plotting training reconstructions...', file=sys.stdout, flush=True)
    fig, axs = plt.subplots(5, 4, figsize=(20, 15))
    for i in range(4):
        for j in range(5):
            axs[j, i].plot(x['x'][i, j].cpu().detach().numpy())
            axs[j, i].plot(x['x_hat'][i, j].cpu().detach().numpy())
            axs[j, i].set_title(f"Sample {i}, Channel {j}")
    plt.tight_layout()
    figure_results['results/recon'] = wandb.Image(fig)
    plt.close(fig)
    #%%
    print('Running PCA...', file=sys.stdout, flush=True)
    subject_pca = PCA(n_components=2)
    task_pca = PCA(n_components=2)
    
    subject_pca = subject_pca.fit_transform(subject_latents)
    task_pca = task_pca.fit_transform(task_latents)
    #%%
    print('Plotting PCA...', file=sys.stdout, flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    plot_latents(fig, ax[0], subject_pca, subjects, tasks, test_loader, which='subject')
    plot_latents(fig, ax[1], task_pca, subjects, tasks, test_loader, which='task')
    ax[0].set_title('Subject PCA')
    ax[1].set_title('Task PCA')
    ax[1].legend()
    plt.tight_layout()
    figure_results['results/pca'] = wandb.Image(fig)
    plt.close(fig)
    
    #%%
    print('Running TSNE...', file=sys.stdout, flush=True)
    subject_tsne, task_tsne = split_do_tsne(subject_latents, task_latents)
    #%%
    print('Plotting TSNE...', file=sys.stdout, flush=True)
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    plot_latents(fig, ax[0, 0], subject_tsne, subjects, tasks, which='subject')
    ax[0, 0].set_title('TSNE: Subject latent colored by subject')
    ax[0, 0].legend()
    plot_latents(fig, ax[1, 0], task_tsne, subjects, tasks, which='task')
    ax[1, 0].set_title('TSNE: Task latent colored by task')
    ax[1, 0].legend()
    plot_latents(fig, ax[0, 1], subject_tsne, subjects, tasks, which='task')
    ax[0, 1].set_title('TSNE: Subject latent colored by task')
    ax[0, 1].legend()
    plot_latents(fig, ax[1, 1], task_tsne, subjects, tasks, which='subject')
    ax[1, 1].set_title('TSNE: Task latent colored by subject')
    ax[1, 1].legend()
    plt.tight_layout()
    figure_results['results/tsne'] = wandb.Image(fig)
    plt.close(fig)
    
    #%%
    #Plot confusion matrix
    print('Plotting confusion matrix...', file=sys.stdout, flush=True)
    subject_cm = test_results['XGB/' + test_loader.split + '/' + 'subject/cm']
    task_cm = test_results['XGB/' + test_loader.split + '/' + 'task/cm']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs.flatten()
    for i, which in enumerate(['subject', 'task', 'paradigm']):
        display_labels = []
        if which == 'subject':
            cm = subject_cm / subject_cm.sum(axis=1, keepdims=True)
            display_labels = [f'S{s}' for s in test_loader.unique_subjects]
        elif which == 'task':
            cm = task_cm / task_cm.sum(axis=1, keepdims=True)
            display_labels = [f'{test_loader.task_to_label[t]}' for t in test_loader.unique_tasks]
        else:
            cm = task_cm[::2,::2] + task_cm[1::2,::2] + task_cm[::2,1::2] + + task_cm[1::2,1::2]
            cm = cm / cm.sum(axis=1, keepdims=True)
            display_labels = [f'{test_loader.task_to_label[t].split("/")[0]}' for t in test_loader.unique_tasks[::2]]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=display_labels)
        disp.plot(ax=ax[i], xticks_rotation='vertical', cmap='Blues', values_format='.2f', text_kw={'fontsize': 7} if which == 'task' else None)
        disp.ax_.get_images()[0].set_clim(0, 1)
        if which == 'subject':
            acc = test_results['XGB/test/subject/balanced_accuracy']
        elif which == 'task':
            acc = test_results['XGB/test/task/balanced_accuracy']
        else:
            acc = test_results['XGB/test/task/paradigm_wise_accuracy']
        ax[i].set_title(f'{which.capitalize()}\nBalanced Accuracy: {100*acc:.2f}%', fontsize=12)
    plt.tight_layout()
    figure_results['results/cm_fig'] = wandb.Image(fig)
    plt.close(fig)
    plt.close('all')
    print('Uploading to wandb', file=sys.stdout, flush=True)
    wandb.log(figure_results)
    print('Done!', file=sys.stdout, flush=True)
    
    if args.save_model:
        save_as = args.model_save_dir+f"{wandb.run.id}-{wandb.run.name}.pt"
        torch.save(model.state_dict(), save_as)
        print(f"Saved model to {save_as}", file=sys.stdout, flush=True)