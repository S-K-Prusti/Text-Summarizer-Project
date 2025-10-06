from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelEvaluationConfig
import evaluate
import torch
import pandas as pd
from tqdm import tqdm

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    @staticmethod
    def generate_batch_sized_chunks(list_of_elements, batch_size):
        """Split the dataset into smaller batches for efficient processing."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    @staticmethod
    def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                                    batch_size=16, device="cpu",
                                    column_text="article",
                                    column_summary="highlights"):
        """Calculate ROUGE metric on the test dataset."""
        article_batches = list(ModelEvaluation.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(ModelEvaluation.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches), desc="Evaluating"
        ):
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)

            summaries = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                length_penalty=0.8,
                num_beams=8,
                max_length=128,
                early_stopping=True
            )

            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for s in summaries
            ]

            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_bart = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = evaluate.load("rouge")

        # ✅ Run evaluation on a small subset for quick testing
        score = ModelEvaluation.calculate_metric_on_test_ds(
            dataset=dataset_samsum_pt['test'][:100],
            metric=rouge_metric,
            model=model_bart,
            tokenizer=tokenizer,
            batch_size=2,
            device=device,
            column_text='dialogue',
            column_summary='summary'
        )

        # ✅ Extract f-measure for each ROUGE score
        rouge_dict = {rn: score[rn] for rn in rouge_names}

        # ✅ Save to CSV
        df = pd.DataFrame(rouge_dict, index=['bart'])
        df.to_csv(self.config.metric_file_name, index=False)