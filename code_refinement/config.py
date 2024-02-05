import argparse

def parse_args(parser):
    parser.add_argument("--model_name",
                        "-m",
                        type=str,
                        default="Salesforce/codet5-base")
    parser.add_argument(
        "--train_file",
        "-f",
        type=str,
        default="data/ref-train.jsonl",
        help="Path to the file that contains the train dataset.")
    parser.add_argument(
        "--valid_file",
        type=str,
        default="data/ref-valid.jsonl",
        help="Path to the file that contains the validation dataset.")
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/ref-test.jsonl",
        help="Path to the file that contains the test dataset.")
    parser.add_argument("--continue_from_checkpoint", action='store_true')
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument(
        "--checkpoint_folder",
        default="output/checkpoints-150000/",
        type=str,
        help=
        "Path to the checkpoint folder. Useful only if continue_from_checkpoint==True."
    )
    parser.add_argument("--model_file",
                        type=str,
                        default="model_state.bin",
                        help="Name of the model's file.")
    parser.add_argument("--optimizer_file",
                        type=str,
                        default="optimizer.pt",
                        help="Name of the optimizer's file.")
    parser.add_argument("--scheduler_file",
                        type=str,
                        default="scheduler.pt",
                        help="Name of the scheduler's file.")
    parser.add_argument("--config_file",
                        type=str,
                        default="config.json",
                        help="Name of the config's file.")
    parser.add_argument("--num_epochs", "-e", type=int, default=30)
    parser.add_argument(
        "--output_dir",
        "-o",
        default="output/",
        type=str,
        help="Path to the output directory where the checkpoints are saved.")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--learning_rate", "-l", type=float, default=3e-4)
    parser.add_argument("--gradient_accumulation_steps",
                        "-g",
                        type=int,
                        default=5)
    parser.add_argument("--seed", "-s", type=int, default=12345)
    parser.add_argument("--beam_size", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=300000)
    parser.add_argument("--log-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=5000)

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    print(args)
