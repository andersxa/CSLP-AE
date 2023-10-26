
import numpy as np
import torch
from torch.nn import functional as F
from collections import defaultdict

@torch.inference_mode(True)
def sample_latents(loader, subject_latents, task_latents, target_subject, target_task, n=2000, specific_task='all', specific_subject='all'):
    if isinstance(specific_subject, int):
        convert_task_latents = task_latents[(loader.subjects == specific_subject) & (loader.tasks == target_task)]
    elif specific_subject == 'all':
        convert_task_latents = task_latents[loader.tasks == target_task]
    elif specific_subject == 'same':
        convert_task_latents = task_latents[(loader.subjects == target_subject) & (loader.tasks == target_task)]
    elif specific_subject == 'different':
        convert_task_latents = task_latents[(loader.subjects != target_subject) & (loader.tasks == target_task)]
    else:
        raise ValueError('specific_subject must be one of [#subject_class_label#, all, same, different]')
    if isinstance(specific_task, int):
        convert_subject_latents = subject_latents[(loader.subjects == target_subject) & (loader.tasks == specific_task)]
    elif specific_task == 'all':
        convert_subject_latents = subject_latents[loader.subjects == target_subject]
    elif specific_task == 'same':
        convert_subject_latents = subject_latents[(loader.subjects == target_subject) & (loader.tasks == target_task)]
    elif specific_task == 'different':
        convert_subject_latents = subject_latents[(loader.subjects == target_subject) & (loader.tasks != target_task)]
    else:
        raise ValueError('specific_task must be one of [#task_class_label#, all, same, different]')
    
    num_task_latents = convert_task_latents.shape[0]
    if num_task_latents < n:
        task_permute_idxs = np.random.randint(0, num_task_latents, size=n)
    else:
        task_permute_idxs = np.random.permutation(num_task_latents)[:n]
    convert_task_latents = convert_task_latents[task_permute_idxs]
    
    num_subject_latents = convert_subject_latents.shape[0]
    if num_subject_latents < n:
        subject_permute_idxs = np.random.randint(0, num_subject_latents, size=n)
    else:
        subject_permute_idxs = np.random.permutation(num_subject_latents)[:n]
    convert_subject_latents = convert_subject_latents[subject_permute_idxs]
    
    return convert_subject_latents, convert_task_latents


@torch.inference_mode(True)
def reconstruct(model, convert_subject_latents, convert_task_latents, batch_size=2048):
    num_latents = convert_subject_latents.shape[0]
    num_batches = int(np.ceil(num_latents / batch_size))
    convert_subject_latents = torch.unflatten(torch.tensor(convert_subject_latents, device='cuda'), 1, (model.latent_dim, model.latent_seqs))
    convert_task_latents = torch.unflatten(torch.tensor(convert_task_latents, device='cuda'), 1, (model.latent_dim, model.latent_seqs))
    reconstructions = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        x_dict = {'s': convert_subject_latents[start_idx:end_idx], 't': convert_task_latents[start_idx:end_idx]}
        x_dict = model.get_x_hat(x_dict)
        reconstructions.append(x_dict['x_hat'])
    return torch.cat(reconstructions, dim=0)

@torch.inference_mode(True)
def get_reconstructed_erps(model, loader, subject_latents, task_latents, t_spec, s_spec, target_subject, target_task1, target_task2, n):
    
    para_label_names = {
        0: 'ERN',
        1: 'ERN',
        2: 'LRP',
        3: 'LRP',
        4: 'MMN',
        5: 'MMN',
        6: 'N2pc',
        7: 'N2pc',
        8: 'N170',
        9: 'N170',
        10: 'N400',
        11: 'N400',
        12: 'P3',
        13: 'P3',
    }

    baseline = defaultdict(lambda: (-0.2,0))
    baseline['LRP'] = (-0.8,-0.6)
    baseline['ERN'] = (-0.4,-0.2)
    epoch_window = defaultdict(lambda: (-0.2, 0.8))
    epoch_window['LRP'] = (-0.8, 0.2)
    epoch_window['ERN'] = (-0.6, 0.4)
    
    baseline1_start, baseline1_end = baseline[para_label_names[target_task1]]
    
    baseline1_start = baseline1_start - epoch_window[para_label_names[target_task1]][0]
    baseline1_end = baseline1_end - epoch_window[para_label_names[target_task1]][0]
    
    baseline2_start, baseline2_end = baseline[para_label_names[target_task2]]
    baseline2_start = baseline2_start - epoch_window[para_label_names[target_task2]][0]
    baseline2_end = baseline2_end - epoch_window[para_label_names[target_task2]][0]
    
    baseline1_start = int(baseline1_start * model.time_resolution)
    baseline1_end = int(baseline1_end * model.time_resolution)
    baseline2_start = int(baseline2_start * model.time_resolution)
    baseline2_end = int(baseline2_end * model.time_resolution)
    
    convert_subject_latents, convert_task_latents = sample_latents(loader, subject_latents, task_latents, target_subject, target_task1, n=n, specific_task=t_spec, specific_subject=s_spec)
    reconstructions = reconstruct(model, convert_subject_latents, convert_task_latents)
    reconstructions = reconstructions*loader.data_std + loader.data_mean
    baseline = reconstructions[:, :, baseline1_start:baseline1_end].mean(2, keepdim=True)
    reconstructions = reconstructions - baseline
    reconstructed_erp1 = reconstructions.mean(0)
    convert_subject_latents, convert_task_latents = sample_latents(loader, subject_latents, task_latents, target_subject, target_task2, n=n, specific_task=t_spec, specific_subject=s_spec)
    reconstructions = reconstruct(model, convert_subject_latents, convert_task_latents)
    reconstructions = reconstructions*loader.data_std + loader.data_mean
    baseline = reconstructions[:, :, baseline2_start:baseline2_end].mean(2, keepdim=True)
    reconstructions = reconstructions - baseline
    reconstructed_erp2 = reconstructions.mean(0)
    return reconstructed_erp1, reconstructed_erp2

@torch.inference_mode(True)
def get_conversion_results(model, loader, subject_latents, task_latents, target_subject, target_task1, target_task2, channel, n):
    real_erp1 = loader.data[(loader.subjects == target_subject) & (loader.tasks == target_task1)]
    real_erp1 = real_erp1.cuda() * loader.data_std + loader.data_mean
    real_erp1 = real_erp1.mean(0)
    real_erp2 = loader.data[(loader.subjects == target_subject) & (loader.tasks == target_task2)]
    real_erp2 = real_erp2.cuda() * loader.data_std + loader.data_mean
    real_erp2 = real_erp2.mean(0)
    recon_erp_ss1, recon_erp_ss2 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'same', 'same', target_subject, target_task1, target_task2, n)
    recon_erp_sd1, recon_erp_sd2 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'same', 'different', target_subject, target_task1, target_task2, n)
    recon_erp_ds1, recon_erp_ds2 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'different', 'same', target_subject, target_task1, target_task2, n)
    recon_erp_dd1, recon_erp_dd2 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'different', 'different', target_subject, target_task1, target_task2, n)
    ss_mse = F.mse_loss(recon_erp_ss1[channel], real_erp1[channel]) + F.mse_loss(recon_erp_ss2[channel], real_erp2[channel])
    sd_mse = F.mse_loss(recon_erp_sd1[channel], real_erp1[channel]) + F.mse_loss(recon_erp_sd2[channel], real_erp2[channel])
    ds_mse = F.mse_loss(recon_erp_ds1[channel], real_erp1[channel]) + F.mse_loss(recon_erp_ds2[channel], real_erp2[channel])
    dd_mse = F.mse_loss(recon_erp_dd1[channel], real_erp1[channel]) + F.mse_loss(recon_erp_dd2[channel], real_erp2[channel])
    
    return ss_mse, sd_mse, ds_mse, dd_mse

@torch.inference_mode(True)
def get_full_conversion_results(model, test_loader, subject_latents, task_latents, N):
    channel_names = [
        'FP1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'PO7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz', 'FP2', 'Fz', 'F4', 'F8', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'PO8', 'PO4', 'O2', 'HEOG', 'VEOG'
    ]
    channel_to_idx = {c: i for i, c in enumerate(channel_names)}
    task_to_channel = {0: 'FCz', 1: 'FCz', 2: 'FCz', 3: 'FCz', 4: 'FCz', 5: 'FCz', 6: 'PO7', 7: 'PO7', 10: 'CPz', 11: 'CPz', 12: 'Pz', 13: 'Pz'}
    ss_mses, sd_mses, ds_mses, dd_mses = [], [], [], []
    mse_results = {}
    for target_subject in test_loader.unique_subjects:
        for target_task1, target_task2 in test_loader.paradigms:
            channel = channel_to_idx[task_to_channel[target_task1]]
            ss_mse, sd_mse, ds_mse, dd_mse = get_conversion_results(model, test_loader, subject_latents, task_latents, target_subject, target_task1, target_task2, channel, N)
            paradigm_name = test_loader.task_to_label[target_task1].split('/')[0]
            mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ss'] = ss_mse
            mse_results[f'MSE/test/{paradigm_name}/{target_subject}/sd'] = sd_mse
            mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ds'] = ds_mse
            mse_results[f'MSE/test/{paradigm_name}/{target_subject}/dd'] = dd_mse
            ss_mses.append(ss_mse)
            sd_mses.append(sd_mse)
            ds_mses.append(ds_mse)
            dd_mses.append(dd_mse)
    #Calculate per subject mean
    for target_subject in test_loader.unique_subjects:
        ss_results, sd_results, ds_results, dd_results = [], [], [], []
        for target_task1, target_task2 in test_loader.paradigms:
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
    for target_task1, target_task2 in test_loader.paradigms:
        paradigm_name = test_loader.task_to_label[target_task1].split('/')[0]
        ss_results, sd_results, ds_results, dd_results = [], [], [], []
        for target_subject in test_loader.unique_subjects:
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
    return mse_results