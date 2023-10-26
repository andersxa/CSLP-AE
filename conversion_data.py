#%%
import wandb
import torch
from split_model import SplitLatentModel
from utils import get_split_latents, CustomLoader
from conversion_utils import get_conversion_results
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--wandb_project', type=str, default='')
parser.add_argument('--model_names', type=str, default='')
parser.add_argument('--conversion_N', type=int, default=2000)
parser.add_argument('--per_errors', type=int, default=1)
args, unknown = parser.parse_known_args()


if __name__ == '__main__':
    model_names = args.model_names.split(',')
    print("Loading data...")
    with torch.inference_mode():
        data_dict = torch.load(args.data_dir+f"simple_data.pt")
        data_dict["data"] = (data_dict["data"] - data_dict["data_mean"]) / data_dict["data_std"]
        test_loader = CustomLoader(data_dict, split='test')
        data_mean = data_dict["data_mean"].detach().clone().contiguous().cuda()
        data_std = data_dict["data_std"].detach().clone().contiguous().cuda()
        del data_dict

    unique_subjects = test_loader.unique_subjects
    unique_tasks = test_loader.unique_tasks

    channel_names = [
        'FP1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'PO7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz', 'FP2', 'Fz', 'F4', 'F8', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'PO8', 'PO4', 'O2', 'HEOG', 'VEOG'
    ]
    channel_to_idx = {c: i for i, c in enumerate(channel_names)}
    #Best channels for each paradigm (according to ERP Core)
    task_to_channel = {0: 'FCz', 1: 'FCz', 2: 'FCz', 3: 'FCz', 4: 'FCz', 5: 'FCz', 6: 'PO7', 7: 'PO7', 10: 'CPz', 11: 'CPz', 12: 'Pz', 13: 'Pz'}

    paradigms = [[0, 1], [2, 3], [4, 5], [6, 7], [10, 11], [12, 13]]

    N = args.conversion_N

    for model_name in tqdm(model_names):
        wandb_id = model_name.split('-')[0]
        with wandb.init(project=args.wandb_project, id=wandb_id, resume='must') as run:
            state_dict = torch.load(args.data_dir + model_name + '.pt')
            model = SplitLatentModel(30, 256, 64, 4, 4, recon_type='mse', content_cosine=1).cuda()
            model.load_state_dict(state_dict)
            model.eval()
            with torch.inference_mode():
                subject_latents, task_latents, subjects, tasks, runs, losses = get_split_latents(model, test_loader, test_loader.get_dataloader(batch_size=model.batch_size, random_sample=False))
                ss_mses, sd_mses, ds_mses, dd_mses = [], [], [], []
                mse_results = {}
                for target_subject in unique_subjects:
                    for target_task1, target_task2 in paradigms:
                        channel = channel_to_idx[task_to_channel[target_task1]]
                        ss_mse, sd_mse, ds_mse, dd_mse = get_conversion_results(model, test_loader, subject_latents, task_latents, target_subject, target_task1, target_task2, channel, N)
                        paradigm_name = test_loader.task_to_label[target_task1].split('/')[0]
                        #mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ss'] = ss_mse
                        #mse_results[f'MSE/test/{paradigm_name}/{target_subject}/sd'] = sd_mse
                        #mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ds'] = ds_mse
                        #mse_results[f'MSE/test/{paradigm_name}/{target_subject}/dd'] = dd_mse
                        ss_mses.append(ss_mse)
                        sd_mses.append(sd_mse)
                        ds_mses.append(ds_mse)
                        dd_mses.append(dd_mse)
                #Calculate per subject mean
                if args.per_errors:
                    for target_subject in unique_subjects:
                        ss_results, sd_results, ds_results, dd_results = [], [], [], []
                        for target_task1, target_task2 in paradigms:
                            paradigm_name = test_loader.task_to_label[target_task1].split('/')[0]
                            ss_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ss'])
                            sd_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/sd'])
                            ds_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ds'])
                            dd_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/dd'])
                        mse_results[f'MSE/test/mean/{target_subject}/ss'] = torch.mean(torch.stack(ss_results))
                        mse_results[f'MSE/test/mean/{target_subject}/sd'] = torch.mean(torch.stack(sd_results))
                        mse_results[f'MSE/test/mean/{target_subject}/ds'] = torch.mean(torch.stack(ds_results))
                        mse_results[f'MSE/test/mean/{target_subject}/dd'] = torch.mean(torch.stack(dd_results))
                #Calculate per paradigm mean
                if args.per_errors:
                    for target_task1, target_task2 in paradigms:
                        paradigm_name = test_loader.task_to_label[target_task1].split('/')[0]
                        ss_results, sd_results, ds_results, dd_results = [], [], [], []
                        for target_subject in unique_subjects:
                            ss_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ss'])
                            sd_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/sd'])
                            ds_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ds'])
                            dd_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/dd'])
                        mse_results[f'MSE/test/mean/{paradigm_name}/ss'] = torch.mean(torch.stack(ss_results))
                        mse_results[f'MSE/test/mean/{paradigm_name}/sd'] = torch.mean(torch.stack(sd_results))
                        mse_results[f'MSE/test/mean/{paradigm_name}/ds'] = torch.mean(torch.stack(ds_results))
                        mse_results[f'MSE/test/mean/{paradigm_name}/dd'] = torch.mean(torch.stack(dd_results))
                
                #Calculate overall mean
                mse_results['MSE/test/mean/ss'] = torch.mean(torch.stack(ss_mses))
                mse_results['MSE/test/mean/sd'] = torch.mean(torch.stack(sd_mses))
                mse_results['MSE/test/mean/ds'] = torch.mean(torch.stack(ds_mses))
                mse_results['MSE/test/mean/dd'] = torch.mean(torch.stack(dd_mses))
                run.log(mse_results)