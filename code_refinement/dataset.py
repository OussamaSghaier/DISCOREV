from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, RobertaTokenizer
from multiprocessing import Pool
import multiprocessing
import json

def custom_collate(batch):
    return batch


def get_dataset(file_path, tokenizer, args, evaluation=False):
    dataset = CodeReviewDataset(file_path, tokenizer, args)
    dataloader = DataLoader(dataset,
                            batch_size=args["batch_size"],
                            shuffle=True,
                            collate_fn=custom_collate)
    return dataset, dataloader


class CodeReviewFeatures:

    def __init__(self, example_id, source_ids, target_ids):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids


class CodeReviewDataset(Dataset):

    def __init__(self, file_path, tokenizer, args, samplenum=-1):
        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        self.tokenizer = tokenizer
        self.args = args
        print("Reading examples from {}".format(file_path))
        examples = [json.loads(line) for line in open(file_path)]
        for i in range(len(examples)):
            if "id" not in examples[i]:
                examples[i]["id"] = i
        if samplenum > 0:
            examples = examples[:samplenum]
        print(f"Tokenize examples: {file_path}")
        self.data = pool.map(self.tokenize, \
            [(example, tokenizer, args) for example in examples])

    def tokenize(self, item):
        example, tokenizer, args = item
        oldlines = example["old"].split("\n")
        newlines = example["new"].split("\n")
        oldlines = [line[1:].strip() for line in oldlines]
        newlines = [line[1:].strip() for line in newlines]
        oldlines = "\n".join(oldlines)
        newlines = "\n".join(newlines)
        oldlines = "<add>" + oldlines.replace("\n", "<add>")
        newlines = "<add>" + newlines.replace("\n", "<add>")
        comment = example["comment"]
        srcids = self.encode_remove(tokenizer, oldlines, args)
        srcids += [tokenizer.msg_id] + self.encode_remove(
            tokenizer, comment, args)
        tgtids = self.encode_remove(tokenizer, newlines, args)
        srcids, tgtids = self.pad_assert(srcids, tgtids, args, tokenizer)
        return CodeReviewFeatures(example["id"], srcids, tgtids)

    @staticmethod
    def process_pred_gold(pred, gold):
        gold = gold.split("\n")
        gold = [line[1:].strip() for line in gold]
        gold = " ".join(gold)
        pred = " ".join(pred.split())
        pred = pred.replace("<add> ", "")
        return pred, gold

    def pad_assert(self, source_ids, target_ids, args, tokenizer):
        source_ids = source_ids[:args['max_source_length'] - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args['max_source_length'] - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        target_ids = target_ids[:args['max_target_length'] - 2]
        target_ids = [tokenizer.bos_id] + target_ids + [tokenizer.eos_id]
        pad_len = args['max_target_length'] - len(target_ids)
        target_ids += [tokenizer.pad_id] * pad_len
        assert len(
            source_ids) == args['max_source_length'], "Not equal length."
        assert len(
            target_ids) == args['max_target_length'], "Not equal length."
        return source_ids, target_ids

    def encode_remove(self, tokenizer, text, args):
        text = tokenizer.encode(text,
                                max_length=args['max_source_length'],
                                truncation=True)
        if type(tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(tokenizer) == RobertaTokenizer:
            return text[1:-1]
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
