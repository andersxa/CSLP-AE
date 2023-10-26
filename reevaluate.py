import sys
import wandb
from split_model import SplitLatentModel
import torch
import numpy as np
from utils import get_results, get_split_latents, split_do_tsne, CustomLoader
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ids', type=str, default='')
parser.add_argument('--data_line', type=str, default='simple')
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--model_save_dir', type=str, default='./')
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--wandb_directory', type=str, default='converting-erps/')

args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    ids = args.ids.split(',')
    api = wandb.Api()
    runs = [api.run(args.wandb_directory + id) for id in ids]
    with torch.no_grad():
        data_dict = torch.load(args.data_dir+f"{args.data_line}_data.pt")
        data_dict["data"] = (data_dict["data"] - data_dict["data_mean"]) / data_dict["data_std"]
        loader = CustomLoader(data_dict, split='train')
        test_loader = CustomLoader(data_dict, split='test')
        del data_dict

    for run in tqdm(runs):
        SEED = run.config['seed']   
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        IN_CHANNELS = run.config['in_channels']
        NUM_LAYERS = run.config['num_layers']
        KERNEL_SIZE = 4
        CHANNELS = run.config['channels']
        LATENT_DIM = run.config['latent_dim']
        RECON_TYPE = run.config['recon_type']
        
        model = SplitLatentModel(IN_CHANNELS, CHANNELS, LATENT_DIM, NUM_LAYERS, KERNEL_SIZE, recon_type=RECON_TYPE)
        
        losses = run.config['losses']
        
        model.set_losses(
            batch_size=run.config['batch_size'],
            losses=losses,
            loader=loader,
        )
        state_dict = torch.load(f"{args.model_save_dir}{run.id}-{run.name}.pt")
        model.load_state_dict(state_dict)
        data_out = {}
        model = model.cuda()
        model.eval()
        with torch.inference_mode(True):
            model.loader = loader
            print(f'Getting train latents {run.name}', file=sys.stdout, flush=True)
            subject_latents, task_latents, subjects, tasks, runs, losses = get_split_latents(model, loader, loader.get_dataloader(batch_size=model.batch_size, random_sample=False))
            print(f'Getting train tsne {run.name}', file=sys.stdout, flush=True)
            subject_tsne, task_tsne = split_do_tsne(subject_latents, task_latents)
            data_out['subject_tsne'] = subject_tsne
            data_out['task_tsne'] = task_tsne
            model.loader = test_loader
            print(f'Getting test latents {run.name}', file=sys.stdout, flush=True)
            subject_latents, task_latents, subjects, tasks, runs, losses = get_split_latents(model, test_loader, test_loader.get_dataloader(batch_size=model.batch_size, random_sample=False))
            print(f'Getting test tsne {run.name}', file=sys.stdout, flush=True)
            subject_tsne, task_tsne = split_do_tsne(subject_latents, task_latents)
            data_out['subject_tsne_test'] = subject_tsne
            data_out['task_tsne_test'] = task_tsne
            print(f'Getting test results {run.name}', file=sys.stdout, flush=True)
            test_results = get_results(subject_latents, task_latents, subjects, tasks, split=test_loader.split, off_class_accuracy=False)
            subject_cm = test_results['XGB/' + test_loader.split + '/' + 'subject/cm']
            task_cm = test_results['XGB/' + test_loader.split + '/' + 'task/cm']
            data_out['subject_cm'] = subject_cm
            data_out['task_cm'] = task_cm
            data_out['test_results'] = test_results
            torch.save(data_out, f"{args.save_dir}/{run.name}_result_data.pt")