import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from dataset import get_dataset
from models import get_models
from evaluation import eval_bleu_epoch_2
import torch

torch.cuda.empty_cache()
import torch.nn as nn
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

    # Load and pre-process valid dataset
    valid_dataset, valid_dataloader = get_dataset(args["valid_file"],
                                                  tokenizer, args)

    steps = args["steps"]
    # Training loop
    print("*** Running evaluation ***")
    bleu = eval_bleu_epoch_2(args, valid_dataset, valid_dataloader, model,
                             tokenizer)
    print(f"> Step: {steps} - BLEU: {bleu}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    main(args)
