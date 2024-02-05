import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from dataset import get_dataset
from models import get_models
from evaluation import eval_bleu_epoch
from models import save_model
import torch

torch.cuda.empty_cache()
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_scheduler
from tqdm.auto import tqdm
import itertools
import argparse
from config import parse_args



def main(args):
    torch.manual_seed(args["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("*** Loading models ***")

    # create model
    config, model, tokenizer = get_models(device, args)

    # deploy on gpu
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    print("*** Loading data ***")

    # Load and pre-process train dataset
    train_dataset, train_dataloader = get_dataset(args["train_file"],
                                                  tokenizer, args)

    # Load and pre-process valid dataset
    valid_dataset, valid_dataloader = get_dataset(args["test_file"], tokenizer,
                                                  args)

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=1e-5)  # 3e-4)

    num_epochs = args["num_epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    if args["continue_from_checkpoint"]:
        optimizer.load_state_dict(
            torch.load(
                os.path.join(args["checkpoint_folder"],
                             args["optimizer_file"])))
        lr_scheduler.load_state_dict(
            torch.load(
                os.path.join(args["checkpoint_folder"],
                             args["scheduler_file"])))

    # Training loop
    i = 0
    steps = 0
    if args["continue_from_checkpoint"]:
        steps = args["steps"]
        print(f"*** Skipping the first {steps} steps")
    print("*** Running training ***")
    progress_bar = tqdm(range(num_training_steps), miniters=1)
    progress_bar.update(steps)
    for epoch in range(num_epochs):
        print(epoch, num_epochs, steps, i)
        for batch in train_dataloader:
            if i < steps:
                i += 1
                continue
            total_loss, loss1, loss2, _, _ = model(batch)
            # update
            optimizer.zero_grad()
            total_loss.mean().backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            steps += 1
            i += 1

            if steps % args["eval_steps"] == 0:
                print("*** Running evaluation ***")
                bleu = eval_bleu_epoch(args, valid_dataset, valid_dataloader,
                                       model, tokenizer)
                print(f"> Epoch: {epoch} - Step: {steps} - BLEU: {bleu}")
                print("*** Running training ***")
                model.train()

            if steps % args["save_steps"] == 0:
                print("*** Saving model ***")
                output_dir = os.path.join(args["output_dir"],
                                          "checkpoints-" + str(steps))
                save_model(model, optimizer, lr_scheduler, output_dir, config)
                print(
                    f"> Epoch: {epoch} - Step: {steps} - model saved to {output_dir}"
                )

            if steps % args["log_steps"] == 0:
                print(
                    f"> Epoch: {epoch} - Step: {steps} - mTotal loss: {total_loss.mean().item()} - Loss1: {loss1.mean().item()} - Loss2: {loss2.mean().item()}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    main(args)
