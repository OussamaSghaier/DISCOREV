from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch

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
    model_name = args['model_name']
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    tokenizer = get_tokenizer(model_name)
    if args['continue_from_checkpoint']:
        model.load_state_dict(
            torch.load(
                os.path.join(args['checkpoint_folder'], args['model_file'])))
        config = AutoConfig.from_pretrained(
            os.path.join(args['checkpoint_folder'], args['config_file']))
    return config, model, tokenizer


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
