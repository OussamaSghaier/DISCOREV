from smooth_bleu import bleu_fromstr
import torch


def preprocess_eval(pred, gold):
    gold = gold.split("\n")
    gold = [line[:].strip() for line in gold]
    gold = " ".join(gold)
    pred = " ".join(pred.split())
    pred = pred.replace("<add> ", "")
    return pred, gold


def eval_bleu_epoch(args, dataset, dataloader, model, tokenizer):
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    pred_ids, ex_ids = [], []
    for _, batch in enumerate(dataloader, start=1):
        ids, src1, src2, tgt1, tgt2 = batch
        source_ids = torch.Tensor(src1).long().to("cuda")
        source_mask = source_ids.ne(tokenizer.pad_id).to("cuda")
        preds = model.model_1.generate(
            source_ids,
            attention_mask=source_mask,
            use_cache=True,
            early_stopping=True,
            max_length=args["max_target_length"],
        )
        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)
        ex_ids.extend(ids)
    pred_nls = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in pred_ids
    ]
    golds = [dataset[i].comment for i in ex_ids]

    for i in range(len(golds)):
        pred_nls[i], golds[i] = preprocess_eval(pred_nls[i], golds[i])

    assert len(pred_ids) == len(ex_ids) == len(pred_nls) == len(golds)

    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    return bleu
