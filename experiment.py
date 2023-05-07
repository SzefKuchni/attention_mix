from bert_mix_utils2 import prepare_dataset, experiment, args_parser

if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(args)

    dataset_name = args.dataset
    tokenizer_name = "bert-base-uncased"
    tokenized_datasets, data_collator = prepare_dataset(dataset_name, tokenizer_name)

    model_name = tokenizer_name

    param_dict = {"num_epochs":args.num_epochs, "lr":args.lr, "approach": args.approach, "dropout": args.dropout, "attention_layer":args.attention_layer, "attention_head":args.attention_head}
    experiment(dataset_name, tokenized_datasets, data_collator, model_name, param_dict, save_model = False, out_dir=args.out_dir)