from smooth_bleu import bleu_fromstr
import torch
from dataset import CodeReviewDataset
import json
from tqdm.auto import tqdm

def eval_bleu_epoch(args, dataset, eval_dataloader, model, tokenizer, device):
    print(f"  ***** Running bleu evaluation on {args['valid_file']} *****")
    print("  Batch size = %d", args['batch_size'])
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    pred_ids, ex_ids = [], []

    num_eval_steps = len(eval_dataloader)
    progress_bar = tqdm(range(num_eval_steps), miniters=1)
    for step, examples in enumerate(eval_dataloader, 1):
        source_ids = torch.tensor([ex.source_ids for ex in examples],
                                  dtype=torch.long).to(device)
        source_mask = source_ids.ne(tokenizer.pad_id)
        preds = model.generate(source_ids,
                               attention_mask=source_mask,
                               use_cache=True,
                               num_beams=args['beam_size'],
                               early_stopping=True,
                               max_length=args['max_target_length'])
        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)
        progress_bar.update(1)
    pred_nls = [
        tokenizer.decode(id,
                         skip_special_tokens=True,
                         clean_up_tokenization_spaces=False) for id in pred_ids
    ]
    valid_file = args['valid_file']
    golds = []
    with open(valid_file, "r") as f:
        for line in f:
            golds.append(json.loads(line)["new"])
    golds = golds[:len(pred_nls)]
    for i in range(len(golds)):
        pred_nls[i], golds[i] = CodeReviewDataset.process_pred_gold(
            pred_nls[i], golds[i])

    em = 0
    for pred, gold in zip(pred_nls, golds):
        if " ".join(pred.split()) == " ".join(gold.split()):
            em += 1
    em = em / len(golds)
    print(f"EM: {em}")
    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    return bleu


def preprocess_eval(pred, gold):
    gold = gold.split("\n")
    gold = [line[:].strip() for line in gold]
    gold = " ".join(gold)
    pred = " ".join(pred.split())
    pred = pred.replace("<add> ", "")
    return pred, gold
