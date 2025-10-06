# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

def _fallback_dataset_from_hub():
    from pathlib import Path

    import pandas as pd
    from huggingface_hub import hf_hub_download

    def load_repo(repo_id: str) -> pd.DataFrame:
        path = hf_hub_download(repo_id, "data/train-00000-of-00001.parquet", repo_type="dataset")
        return pd.read_parquet(path)

    train_df = load_repo("AI-MO/aimo-validation-aime")
    test_df = load_repo("MathArena/aime_2025")

    train_records = [
        {
            "input": row["problem"],
            "additional_context": {"solution": row.get("solution", "")},
            "answer": "### " + str(row.get("answer", "")),
        }
        for _, row in train_df.iterrows()
    ]

    test_records = [
        {"input": row["problem"], "answer": "### " + str(row.get("answer", ""))}
        for _, row in test_df.iterrows()
    ]

    import random

    random.Random(0).shuffle(train_records)
    mid = len(train_records) // 2
    trainset = train_records[:mid]
    valset = train_records[mid:]
    return trainset, valset, test_records * 5


def init_dataset():
    import random

    try:
        from datasets import load_dataset

        train_split = [
            {
                "input": x["problem"],
                "additional_context": {"solution": x["solution"]},
                "answer": "### " + str(x["answer"]),
            }
            for x in load_dataset("AI-MO/aimo-validation-aime")["train"]
        ]
        random.Random(0).shuffle(train_split)
        test_split = [
            {"input": x["problem"], "answer": "### " + str(x["answer"])}
            for x in load_dataset("MathArena/aime_2025")["train"]
        ]

        trainset = train_split[: len(train_split) // 2]
        valset = train_split[len(train_split) // 2 :]
        testset = test_split * 5
        return trainset, valset, testset
    except Exception:
        return _fallback_dataset_from_hub()
