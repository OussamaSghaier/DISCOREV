from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch
from torch import nn
from utils import softargmax, differentiable_round


class CodeReviewer(nn.Module):

    def __init__(self, model_name, device, args, alpha=0.5):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.model_1 = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, config=self.config)
        self.model_2 = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, config=self.config)
        self.tokenizer = get_tokenizer(model_name)
        self.device = device
        self.alpha = alpha
        self.args = args
        self.embed_criterion = nn.MSELoss()
        return


    def forward(self, code_batch):
        device = self.device
        tokenizer = self.tokenizer

        ids, src1, src2, tgt1, tgt2 = code_batch

        # Model 1 forward pass
        src_t1, labels1 = torch.tensor(src1, dtype=torch.long,
                                       device=device), torch.tensor(
                                           tgt1,
                                           dtype=torch.long,
                                           device=device)
        src_mask1 = src_t1.ne(tokenizer.pad_id)
        labels1[labels1 == tokenizer.pad_id] = -100

        outputs1 = self.model_1(input_ids=src_t1,
                                labels=labels1,
                                attention_mask=src_mask1)
        loss1 = outputs1.loss  # loss 1 : scalar
        logits1 = outputs1.logits  # logits1 : (batch_size, gen_len, vocab_size)
        # derivable softargmax instead of argmax
        pred1 = softargmax(logits1, device)  # (batch_size, gen_len, 1)
        pred1 = differentiable_round(pred1)

        # Model 2 forward pass
        src_t2 = torch.tensor(src2, dtype=torch.long, device=device)
        msg_id = torch.tensor(tokenizer.msg_id,
                              dtype=torch.long,
                              device=device).repeat(pred1.size(0), 1)
        inpt_2 = torch.cat((src_t2, msg_id, pred1),
                           dim=1)  # (batch_size, seq_len + 1 + gen_len)
        labels2 = torch.tensor(tgt2, dtype=torch.long, device=device)
        labels2[labels2 == tokenizer.pad_id] = -100
        inpt_mask2 = inpt_2.ne(tokenizer.pad_id)

        outputs2 = self.model_2(input_ids=inpt_2,
                                labels=labels2,
                                attention_mask=inpt_mask2)
        loss2 = outputs2.loss  # loss 2 : scalar
        logits2 = outputs2.logits  # logits2 : (batch_size, gen_len, vocab_size)
        pred2 = torch.argmax(logits2, dim=-1)  # (batch_size, gen_len, 1)

        total_loss = self.alpha * loss1 + (1 - self.alpha) * loss2

        if self.args["compare_embeddings"]:
            last_layer1 = outputs1.encoder_last_hidden_state
            last_layer2 = outputs2.encoder_last_hidden_state
            embed_loss = self.embed_criterion(last_layer1, last_layer2)
            total_loss += embed_loss

        return (
            total_loss,
            loss1,
            loss2,
            pred1,
            pred2,
        )  # pred1.detach().cpu(), pred2.detach().cpu()


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.special_dict = {
        f"<e{i}>": tokenizer.get_vocab()[f"<e{i}>"]
        for i in range(99, -1, -1)
    }
    tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    tokenizer.bos_id = tokenizer.get_vocab()["<s>"]
    tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    tokenizer.eos_id = tokenizer.get_vocab()["</s>"]
    tokenizer.msg_id = tokenizer.get_vocab()["<msg>"]
    tokenizer.keep_id = tokenizer.get_vocab()["<keep>"]
    tokenizer.add_id = tokenizer.get_vocab()["<add>"]
    tokenizer.del_id = tokenizer.get_vocab()["<del>"]
    tokenizer.start_id = tokenizer.get_vocab()["<start>"]
    tokenizer.end_id = tokenizer.get_vocab()["<end>"]
    return tokenizer


def get_models(device, args):
    model = CodeReviewer(args["model_name"], device, args, alpha=0.5)

    if args["continue_from_checkpoint"]:
        model.load_state_dict(
            torch.load(
                os.path.join(args["checkpoint_folder"], args["model_file"])))
        config = AutoConfig.from_pretrained(
            os.path.join(args["checkpoint_folder"], args["config_file"]))

    return model.config, model, model.tokenizer


def save_model(model, optimizer, scheduler, output_dir, config):
    if not os.path.exists(output_dir):
        original_umask = os.umask(0)
        os.makedirs(output_dir, 0o777)
        os.umask(original_umask)
    modelsv = model.module if hasattr(model, "module") else model
    config.save_pretrained(output_dir)
    output_model_file = os.path.join(output_dir, "model_state.bin")
    torch.save(modelsv.state_dict(), output_model_file)
    output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
    torch.save(
        optimizer.state_dict(),
        output_optimizer_file,
        _use_new_zipfile_serialization=False,
    )
    output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
    torch.save(
        scheduler.state_dict(),
        output_scheduler_file,
        _use_new_zipfile_serialization=False,
    )
