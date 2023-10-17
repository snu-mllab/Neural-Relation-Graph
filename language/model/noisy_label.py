if __name__ == '__main__':
    """Get noisy target labels
    """
    import torch
    from math import ceil
    from argument import args
    from .load import get_raw_data, update_check_noise

    base_path = args.cache_dir
    task_name = args.task_name
    size = 'large'
    epoch = 4
    raw_datasets, label_list, num_labels, is_regression = get_raw_data(task_name)

    logit = torch.load(
        f'{base_path}/{task_name}/roberta-{size}_fp16/epoch_{epoch}/feat_train.pt')['logit']
    prob = torch.softmax(logit, dim=-1)

    label = torch.tensor(raw_datasets['train']['label'])
    acc = (prob.max(-1)[1] == label).float().mean()
    print(f"[{task_name}] Train acc: {acc*100:.1f}")

    # noise target
    noise_ratio = 0.1
    idx_clean = None

    step = int(1 / noise_ratio)
    prob_ans = torch.gather(prob, 1, label[:, None]).squeeze()
    if task_name == 'sst2':
        valid = prob_ans > 0.6
        n = int(ceil(valid.sum() * noise_ratio))
        idx_chg = torch.arange(len(label))[valid][::step][:n]

        label_orig = label[idx_chg]
        label_chg = 1 - label_orig
        label[idx_chg] = label_chg
    elif task_name == 'mnli':
        valid = prob_ans > 0.6
        n = int(ceil(valid.sum() * noise_ratio))
        idx_chg = torch.arange(len(label))[valid][::step][:n]

        label_orig = label[idx_chg]
        torch.manual_seed(0)
        label_chg = (label_orig + torch.randint(1, 3, size=label_orig.shape)) % 3
        label[idx_chg] = label_chg

    raw_datasets = update_check_noise(raw_datasets, label.tolist())
    path = f'{base_path}/{task_name}/target{noise_ratio}_{size}_{epoch}.pt'
    torch.save(
        {
            'idx_clean': idx_clean,
            'idx_chg': idx_chg,
            'targets': raw_datasets['train']['label']
        }, path)
