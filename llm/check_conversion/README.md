# Evaluate A Converted Checkpoint

This command will run the model on a `.jsonl` file of text and report cross entropy loss.

Put the data you want in a `.jsonl` file `data/validation.jsonl`. Each entry should be json with 'text' field.

```
{ 'text': 'This is an example entry for input data.' }
```

This script is just a hack altered version of standard training, so there is a 1-line training file so it doesn't complain about there being no training file.

It will set up the trainer, not bother to train, and just run the final evaluation.

```
python evaluate.py --output_dir evaluation_output --do_eval --model_name_or_path models/biomedlm_checkpoint --tokenizer_name models/biomedlm_checkpoint --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --save_strategy no --dataset_dir data --overwrite_output_dir --bf16 --dataset_id pubmed_abstracts --preprocessing_num_workers 1
```
