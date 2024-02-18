from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, RobertaTokenizer
from multiprocessing import Pool
import multiprocessing
import json

def custom_collate(batch):
    ids = [obj.id for obj in batch]
    source1 = [obj.source_ids1 for obj in batch]
    source2 = [obj.source_ids2 for obj in batch]
    target1 = [obj.target_ids1 for obj in batch]
    target2 = [obj.target_ids2 for obj in batch]
    return ids, source1, source2, target1, target2


def get_dataset(file_path, tokenizer, args, evaluation=False):
    dataset = CodeReviewDataset(file_path, tokenizer, args)
    dataloader = DataLoader(dataset,
                            batch_size=args["batch_size"],
                            shuffle=True,
                            collate_fn=custom_collate)
    return dataset, dataloader


class CodeReviewFeatures:

    def __init__(
        self,
        example_id,
        source_ids1,
        source_ids2,
        target_ids1,
        target_ids2,
        source_code,
        target_code,
        comment,
    ):
        self.source_code = source_code
        self.target_code = target_code
        self.comment = comment
        self.id = example_id
        self.source_ids1 = source_ids1
        self.source_ids2 = source_ids2
        self.target_ids1 = target_ids1
        self.target_ids2 = target_ids2


class CodeReviewDataset(Dataset):

    def __init__(self, file_path, tokenizer, args):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.args = args

        print("Reading examples from {}".format(file_path))
        data = [json.loads(line) for line in open(file_path)]
        data = data[0:10]
        print(f"data size: {len(data)}")
        for i in range(len(data)):
            if "id" not in data[i]:
                data[i]["id"] = i

        print(f"Tokenize examples: {file_path}")
        # Tokenize the data
        cpu_count = multiprocessing.cpu_count()
        pool = Pool(cpu_count)
        self.data = pool.map(self.tokenize, data)

    # tokenize, encode, and pad old code and comment for the first model and comment and revised code for the second model
    def tokenize(self, item):
        oldlines = item["old"].split("\n")
        newlines = item["new"].split("\n")
        oldlines = [line[1:].strip() for line in oldlines]
        newlines = [line[1:].strip() for line in newlines]
        oldlines = " ".join(oldlines)
        newlines = " ".join(newlines)
        comment = item["comment"]
        srcids1 = self.encode(oldlines)
        srcids21 = self.encode(oldlines)
        srcids22 = self.encode(comment)
        tgtids1 = [self.tokenizer.msg_id] + self.encode(comment)
        tgtids2 = self.encode(newlines)
        srcids1, srcids2, tgtids1, tgtids2 = self.pad_assert(
            srcids1, tgtids1, tgtids2)

        return CodeReviewFeatures(
            item["id"],
            srcids1,
            srcids2,
            tgtids1,
            tgtids2,
            item["old"],
            item["new"],
            item["comment"],
        )

    def pad_assert(self, source_ids1, target_ids1, target_ids2):
        max_src_len = (self.args["max_source_length"] +
                       self.args["max_target_length"] + 1)
        source_ids1 = source_ids1[:max_src_len]
        source_pad_len1 = max_src_len - len(source_ids1)
        source_ids1 += [self.tokenizer.pad_id] * source_pad_len1

        source_ids2 = source_ids1[:self.args["max_source_length"]]
        source_pad_len2 = self.args["max_source_length"] - len(source_ids2)
        source_ids2 += [self.tokenizer.pad_id] * source_pad_len2

        target_ids1 = target_ids1[:self.args["max_target_length"]]
        target_pad_len1 = self.args["max_target_length"] - len(target_ids1)
        target_ids1 += [self.tokenizer.pad_id] * target_pad_len1

        target_ids2 = target_ids2[:self.args["max_target_length"]]
        target_pad_len2 = self.args["max_target_length"] - len(target_ids2)
        target_ids2 += [self.tokenizer.pad_id] * target_pad_len2

        assert len(source_ids1) == max_src_len, "Not equal length."
        assert len(
            source_ids2) == self.args["max_source_length"], "Not equal length."
        assert len(
            target_ids1) == self.args["max_target_length"], "Not equal length."
        assert len(
            target_ids2) == self.args["max_target_length"], "Not equal length."
        return source_ids1, source_ids2, target_ids1, target_ids2

    def pad_assert2(self, source_ids1, source_ids2):
        source_ids1 = source_ids1[:self.args["max_source_length"]]
        source_pad_len1 = self.args["max_source_length"] - len(source_ids1)
        source_ids1 += [self.tokenizer.pad_id] * source_pad_len1

        source_ids2 = source_ids2[:self.args["max_target_length"]]
        source_pad_len2 = self.args["max_target_length"] - len(source_ids2)
        source_ids2 += [self.tokenizer.pad_id] * source_pad_len2

        return source_ids1 + [self.tokenizer.msg_id] + source_ids2

    def encode(self, text):
        text = self.tokenizer.encode(text,
                                     max_length=self.args["max_source_length"],
                                     truncation=True)
        if type(self.tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(self.tokenizer) == RobertaTokenizer:
            return text[1:-1]
        else:
            return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
