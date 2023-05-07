#import torchmetrics
#import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from datasets import ClassLabel, Value
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
import torch
from tqdm.auto import tqdm
import evaluate
import pickle
import torch.nn.functional as F
from datetime import datetime
import argparse
import random
import copy

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='sst', help="Dataset on which to perform experiment")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-6, help="learning rate")
    parser.add_argument('--approach', type=str, default='base', help="...")
    parser.add_argument('--dropout', type=float, default=0.1, help="classifier dropout")
    parser.add_argument('--attention_layer', type=int, default=999, help="attention layer to be used")
    parser.add_argument('--attention_head', type=int, default=999, help="attention head to be used")
    parser.add_argument('--out_dir', type=str, default='broad_vs_narrow_attn', help="output folder group / experiment name")

    args = parser.parse_args()
    return args
    
def sst_transform_label(example):
    if example["label"] <= 0.2:
        value = 0
    elif 0.2 < example["label"] <= 0.4:
        value = 1
    elif 0.4 < example["label"] <= 0.6:
        value = 2
    elif 0.6 < example["label"] <= 0.8:
        value = 3
    else:
        value = 4
    example["label"] = int(value)
    return example

def prepare_dataset(dataset, tokenizer_name, debug = False):
    raw_datasets = load_dataset(dataset)
    
    if debug:
        print("#################")
        raw_train_dataset = raw_datasets["train"]
        print(raw_train_dataset.features)

    if dataset == "sst":
        raw_datasets = raw_datasets.map(sst_transform_label)
        raw_datasets = raw_datasets.rename_column("sentence", "text")

        for dataset_name in raw_datasets:
            temp = raw_datasets[dataset_name]
            new_features = temp.features.copy()
            new_features["label"] = ClassLabel(names=["0", "1", "2", "3", "4"])
            temp = temp.cast(new_features)
            raw_datasets[dataset_name] = temp

    # if dataset == "trec":
    #     raw_datasets = raw_datasets.rename_column("text", "sentence")

    if debug:
        print("#################")
        raw_train_dataset = raw_datasets["train"]
        print(raw_train_dataset.features)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    if debug:
        print("#################")
        raw_train_dataset = tokenized_datasets["train"]
        print(raw_train_dataset.features)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    if dataset == "sst":
        tokenized_datasets = tokenized_datasets.remove_columns(["text", "tree","tokens"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
    if dataset == "trec":
        tokenized_datasets = tokenized_datasets.remove_columns(["text", "fine_label"])
        tokenized_datasets = tokenized_datasets.rename_column("coarse_label", "labels")
        tokenized_datasets.set_format("torch")
    if dataset in ["imdb","rotten_tomatoes"]:
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

    if debug:
        print("#################")
        raw_train_dataset = tokenized_datasets["train"]
        print(raw_train_dataset.features)

    if dataset in ["imdb","trec"]:
        new = tokenized_datasets["train"].train_test_split(test_size=0.1)
        tokenized_datasets["train"] = new["train"]
        tokenized_datasets["validation"] = new["test"]

    return tokenized_datasets, data_collator

def experiment(dataset_name, tokenized_datasets, data_collator, model_name, param_dict, save_model=False, out_dir = None):

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=8, collate_fn=data_collator
    )

    if dataset_name == "sst":
        num_labels = 5
    if dataset_name == "trec":
        num_labels = 6
    if dataset_name in ["imdb","rotten_tomatoes"]:
        num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, output_attentions=True, classifier_dropout = param_dict["dropout"])

    optimizer = AdamW(model.parameters(), lr=param_dict["lr"])

    num_training_steps = param_dict["num_epochs"] * len(train_dataloader)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print("device:", device)
    model.to(device)
    #print("model.device:", model.device)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_model_acc = 0
    progress_bar = tqdm(range(num_training_steps))

    cost_fun = torch.nn.CrossEntropyLoss()
    #print("cost_fun.device:", cost_fun.device)
    today = datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')

    for epoch in range(param_dict["num_epochs"]):
        metric = evaluate.load("accuracy")
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            #print("batch['input_ids'].device:", batch["input_ids"].device)

            if param_dict["approach"] == "base":
                embedding = model.bert.embeddings(batch["input_ids"])
                logits = model.classifier(model.dropout(model.bert.pooler(model.bert.encoder(embedding).last_hidden_state)))
                probabilities = F.softmax(logits, dim=-1)
                one_hot_labels = torch.nn.functional.one_hot(batch["labels"], num_classes=num_labels)
                one_hot_labels = one_hot_labels.type(torch.FloatTensor).to(device)
                #print("probabilities.device:", probabilities.device)
                #print("one_hot_labels.device:", one_hot_labels.device)
                loss = cost_fun(probabilities, one_hot_labels)

            if param_dict["approach"] == "mixup_encoding":
                att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][0].mean(axis=[1,2])
                idx = torch.randperm(att1.shape[0])
                att2 = att1[idx]
                mixup_matrix = torch.divide(att1, torch.add(att1, att2)).nan_to_num()
                mixup_matrix = mixup_matrix[:,:,None]
                lam = np.random.beta(1,1)
                mixup_matrix = torch.full_like(mixup_matrix, lam)
                embedding = model.bert.embeddings(batch["input_ids"])
                encoding = model.bert.encoder(embedding).last_hidden_state
                # 3. mix
                mixed_encoding = torch.mul(mixup_matrix, encoding) + torch.mul(1-mixup_matrix, encoding[idx])
                # 4. rest of model
                logits = model.classifier(model.dropout(model.bert.pooler(mixed_encoding)))
                probabilities = F.softmax(logits, dim=-1)
                one_hot_labels = torch.nn.functional.one_hot(batch["labels"], num_classes=num_labels)
                mixup_label = lam
                one_hot_labels = torch.mul(mixup_label, one_hot_labels) + torch.mul(1-mixup_label, one_hot_labels[idx])
                cel = torch.nn.CrossEntropyLoss()
                loss = cost_fun(probabilities, one_hot_labels)
                #########################

            if param_dict["approach"] in ["mixup_embedding","mixup_embedding_att_layer","mixup_embedding_att_layer_head","mixup_embedding_random_layer","mixup_embedding_random_layer_head","mixup_embedding_selected_layers_heads_base","mixup_embedding_selected_layers_heads_best"]:
                # How to use attention
                if param_dict["approach"] == "mixup_embedding":
                    att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][0].mean(axis=[1,2])
                if param_dict["approach"] == "mixup_embedding_att_layer":
                    att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][param_dict["attention_layer"]].mean(axis=[1,2])
                if param_dict["approach"] == "mixup_embedding_att_layer_head":
                    att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][param_dict["attention_layer"]][:,param_dict["attention_head"],:,:].mean(axis=[1])
                if param_dict["approach"] == "mixup_embedding_random_layer":
                    random_layer = random.randint(0, 11)
                    att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][random_layer].mean(axis=[1,2])
                if param_dict["approach"] == "mixup_embedding_random_layer_head":
                    random_layer = random.randint(0, 11)
                    random_head = random.randint(0, 11)
                    att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][random_layer][:,random_head,:,:].mean(axis=[1])
                if param_dict["approach"] == "mixup_embedding_selected_layers_heads_base":
                    if dataset_name == "sst":
                        list_better_than_base = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 8), (2, 9), (2, 10), (10, 0), (10, 3), (10, 11), (11, 2), (11, 3), (11, 7)]
                    if dataset_name == "trec":
                        list_better_than_base = [(0, 0), (0, 2), (0, 4), (0, 5), (0, 6), (0, 7), (0, 9), (0, 11), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (10, 0), (10, 1), (10, 2), (10, 5), (10, 8), (10, 9), (10, 10), (11, 6)]
                    random_layer, random_head = list_better_than_base[random.sample(range(0,len(list_better_than_base),1),1)[0]]
                    att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][random_layer][:,random_head,:,:].mean(axis=[1])
                if param_dict["approach"] == "mixup_embedding_selected_layers_heads_best":
                    if dataset_name == "sst":
                        list_better_than_best = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 11), (1, 0), (1, 2), (1, 5), (1, 11), (2, 0), (2, 1), (2, 2), (2, 8), (2, 10), (10, 3), (11, 2)]
                    if dataset_name == "trec":
                        list_better_than_best = [(0, 2), (1, 0), (1, 2), (1, 5), (1, 7), (1, 11)]
                    random_layer, random_head = list_better_than_best[random.sample(range(0,len(list_better_than_best),1),1)[0]]
                    att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][random_layer][:,random_head,:,:].mean(axis=[1])
                
                idx = torch.randperm(att1.shape[0])
                att2 = att1[idx]
                mixup_matrix = torch.divide(att1, torch.add(att1, att2)).nan_to_num()
                mixup_matrix = mixup_matrix[:,:,None]
                if param_dict["approach"] == "mixup_embedding":
                    lam = np.random.beta(1,1)
                    mixup_matrix = torch.full_like(mixup_matrix, lam)
                embedding = model.bert.embeddings(batch["input_ids"])
                mixed_embedding = torch.mul(mixup_matrix, embedding) + torch.mul(1-mixup_matrix, embedding[idx])
                logits = model.classifier(model.dropout(model.bert.pooler(model.bert.encoder(mixed_embedding).last_hidden_state)))
                probabilities = F.softmax(logits, dim=-1)
                one_hot_labels = torch.nn.functional.one_hot(batch["labels"], num_classes=num_labels)
                if param_dict["approach"] == "mixup_embedding":
                    mixup_label = lam
                else:
                    mixup_label = mixup_matrix.mean(axis=1)
                one_hot_labels = torch.mul(mixup_label, one_hot_labels) + torch.mul(1-mixup_label, one_hot_labels[idx])
                loss = cost_fun(probabilities, one_hot_labels)
                #########################

            # if param_dict["approach"] == "mixup_embedding_att_layer":
            #     att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][param_dict["attention_layer"]].mean(axis=[1,2])
            #     idx = torch.randperm(att1.shape[0])
            #     att2 = att1[idx]
            #     mixup_matrix = torch.divide(att1, torch.add(att1, att2)).nan_to_num()
            #     mixup_matrix = mixup_matrix[:,:,None]
            #     #lam = np.random.beta(1,1)
            #     #mixup_matrix = torch.full_like(mixup_matrix, lam)
            #     embedding = model.bert.embeddings(batch["input_ids"])
            #     mixed_embedding = torch.mul(mixup_matrix, embedding) + torch.mul(1-mixup_matrix, embedding[idx])
            #     logits = model.classifier(model.dropout(model.bert.pooler(model.bert.encoder(mixed_embedding).last_hidden_state)))
            #     probabilities = F.softmax(logits, dim=-1)
            #     one_hot_labels = torch.nn.functional.one_hot(batch["labels"], num_classes=num_labels)
            #     mixup_label = mixup_matrix.mean(axis=1)
            #     one_hot_labels = torch.mul(mixup_label, one_hot_labels) + torch.mul(1-mixup_label, one_hot_labels[idx])
            #     loss = cost_fun(probabilities, one_hot_labels)
            #     #########################

            # if param_dict["approach"] == "mixup_embedding_att_layer_head":
            #     att1 = model.bert.encoder(model.bert.embeddings(batch["input_ids"]), output_attentions = True)["attentions"][param_dict["attention_layer"]][:,param_dict["attention_head"],:,:].mean(axis=[1])
            #     idx = torch.randperm(att1.shape[0])
            #     att2 = att1[idx]
            #     mixup_matrix = torch.divide(att1, torch.add(att1, att2)).nan_to_num()
            #     mixup_matrix = mixup_matrix[:,:,None]
            #     #lam = np.random.beta(1,1)
            #     #mixup_matrix = torch.full_like(mixup_matrix, lam)
            #     embedding = model.bert.embeddings(batch["input_ids"])
            #     mixed_embedding = torch.mul(mixup_matrix, embedding) + torch.mul(1-mixup_matrix, embedding[idx])
            #     logits = model.classifier(model.dropout(model.bert.pooler(model.bert.encoder(mixed_embedding).last_hidden_state)))
            #     probabilities = F.softmax(logits, dim=-1)
            #     one_hot_labels = torch.nn.functional.one_hot(batch["labels"], num_classes=num_labels)
            #     mixup_label = mixup_matrix.mean(axis=1)
            #     one_hot_labels = torch.mul(mixup_label, one_hot_labels) + torch.mul(1-mixup_label, one_hot_labels[idx])
            #     loss = cost_fun(probabilities, one_hot_labels)
            #     #########################

            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            
            #print("loss:", loss)
            #print("type(loss):",type(loss))
            train_loss.append(loss.detach().cpu().numpy())
            loss.backward()

            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        acc = metric.compute()
        #print("acc:", acc)
        #print("type(acc):",type(acc))
        #print("acc['accuracy']:", acc['accuracy'])
        #print("type(acc['accuracy']):",type(acc['accuracy']))
        train_acc.append(acc)

        if save_model:
            if epoch % 100 == 0:
                model.save_pretrained(f"./{out_dir}/{dataset_name}/{model_name}_{param_dict}_{today}/epoch_{epoch}/")
        
        metric = evaluate.load("accuracy")
        model.eval()
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                val_loss.append(loss.detach().cpu().numpy())

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        acc = metric.compute()
        val_acc.append(acc)
        print(acc)

        if acc["accuracy"] > best_model_acc:
            best_model_acc = acc["accuracy"]
            best_model = copy.deepcopy(model)

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = best_model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    test_acc_best = metric.compute()
    print("test_acc - best:", test_acc_best["accuracy"])

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    test_acc_last = metric.compute()
    print("test_acc - last:", test_acc_last["accuracy"])

    pickle.dump({"train_loss":train_loss,
                 "val_loss":val_loss, 
                 "train_acc":train_acc, 
                 "val_acc":val_acc,
                 "test_acc_best":test_acc_best,
                 "test_acc_last":test_acc_last}, 
                 open(f"./{out_dir}/{dataset_name}/{model_name}_{param_dict}_{today}.pkl", "wb"))