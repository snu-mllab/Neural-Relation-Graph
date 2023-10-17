import torch
import os
from math import ceil


def load_feat(path):
    result = torch.load(f"{path}/feat_train.pt")
    feat_train = result['feat_cls'].cuda()
    logit_train = result['logit'].cuda()
    prob_train = torch.softmax(logit_train, dim=-1)

    result = torch.load(f"{path}/feat_validation.pt")
    feat_val = result['feat_cls'].cuda()
    logit_val = result['logit'].cuda()
    prob_val = torch.softmax(logit_val, dim=-1)

    mem = torch.cuda.memory_allocated() / 1024.**2
    print(f"Load features: {mem:.0f}MB", feat_train.shape, feat_val.shape)
    return feat_train, feat_val, prob_train, prob_val


def normalize(feat, nc=50000):
    """Normalize feature

    Args:
        feat (torch tensor, [N,d])
        nc (int, optional): Maximum number of tensor size for each iteration. Defaults to 50000.
    """
    with torch.no_grad():
        split = ceil(len(feat) / nc)
        for i in range(split):
            feat_ = feat[i * nc:(i + 1) * nc]
            feat[i * nc:(i + 1) * nc] = feat_ / torch.sqrt((feat_**2).sum(-1)).reshape(-1, 1)

    return feat


if __name__ == '__main__':
    # Calculate and save features
    import time
    from argument import args
    from model.load import get_processed_data, get_raw_data, get_model
    from transformers import DataCollatorWithPadding
    from torch.utils.data import DataLoader

    torch.cuda.empty_cache()

    name = f'{args.task_name}/{args.model_name}/epoch_{args.epoch}'
    path = os.path.join(args.cache_dir, name)

    # Load model and dataset
    raw_datasets, label_list, num_labels, _ = get_raw_data(args.task_name)
    config, tokenizer, model = get_model(args.task_name, path, num_labels)
    processed_datasets, train_dataset, eval_dataset = get_processed_data(
        args.task_name,
        model,
        config,
        tokenizer,
        raw_datasets,
        label_list,
        num_labels,
    )

    print("\nsample: ", tokenizer.decode(train_dataset['input_ids'][0]),
          label_list[train_dataset['labels'][0]])

    # Calculate and save data features
    model.eval()
    model.cuda()
    for key, dataset in [('validation', eval_dataset), ('train', train_dataset)]:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=data_collator)

        cor = 0
        total = 0
        s = time.time()
        feat_dict = {'feat_cls': [], 'logit': []}
        for i, batch in enumerate(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.no_grad():
                feat_seq, feat, logit, loss = model(**batch, return_feat=True)
                feat_dict['feat_cls'].append(feat_seq[:, 0].cpu())
                feat_dict['logit'].append(logit.cpu())

                pred = logit.max(1)[1]
                cor += (batch['labels'] == pred).sum().item()
                total += len(pred)

            mem = torch.cuda.max_memory_allocated() / 1024.**2
            left = (time.time() - s) / (i + 1) * (len(dataloader) - i - 1)
            print(f"[{key} {left:.0f}s] acc: {cor / total * 100:3.2f}, {mem:.0f} MB", end='\r')

        print(f"{key} acc: {cor / total * 100:3.2f}", " " * 20)

        feat_dict = {k: torch.cat(v) for k, v in feat_dict.items()}
        torch.save(feat_dict, f"{path}/feat_{key}.pt")
