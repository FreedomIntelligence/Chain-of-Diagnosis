#%%
import os
import copy
import json
import torch
import logging
import argparse
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
import transformers
from typing import Sequence
import datasets
import shutil
import json
import random


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import argparse



sampled_ids = set()
class WeightedRandomSampler(Sampler[int]):
    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = False, manual_seed=2147483647) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0 or num_samples > len(weights):
            raise ValueError("num_samples should be a positive integer "
                             "value less than or equal to len(weights), but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        global sampled_ids
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = False
        self.generator = torch.Generator()
        self.generator.manual_seed(manual_seed)
        self.rand_list = torch.multinomial(self.weights, self.weights.shape[0], self.replacement, generator=self.generator).tolist()
        self.pos = 0
        self.sampled_ids = sampled_ids

    def __iter__(self):
        while self.pos < self.num_samples:
            # Avoiding duplicates by checking against sampled_ids set
            idx = self.rand_list[self.pos]
            self.pos += 1
            # if idx not in self.sampled_ids:
            self.sampled_ids.add(idx)
            yield idx

    def __len__(self) -> int:
        return self.num_samples

    def update_dynamic_weight(self, new_weights: Sequence[float]):
        # Making sure the weights are valid
        if len(new_weights) != len(self.weights):
            raise ValueError("Length of new_weights must match the current weights")

        self.weights = torch.as_tensor(new_weights, dtype=torch.double)

        available_indices = list(set(range(len(self.weights))) - self.sampled_ids)
        available_weights = [self.weights[i] for i in available_indices]

        # Resample taking into account already sampled ids
        new_samples = torch.multinomial(torch.as_tensor(available_weights), len(available_indices), self.replacement, generator=self.generator)
        new_list = [available_indices[i] for i in new_samples.tolist()]
        self.pos = len(self.sampled_ids)
        self.rand_list[self.pos:] = new_list
        assert len(self.rand_list) == len(new_weights)

class HuatuoGPT_data(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, dev_ratio = None):
        self.config = config
        self.tokenizer = tokenizer

        with open(config.data_dir) as f:
            data = json.load(f)
        for da in data:
            da['conversations'] = json.loads(da['CoD_conversations'])

        if dev_ratio:
            random.shuffle(data)
            dev_num = int( dev_ratio * len(data))
            eval_data = data[:dev_num]
            data = data[dev_num:]
            with open(os.path.join(config.save_path,'eval.json'), 'w') as fw:
                json.dump(eval_data,fw,ensure_ascii=False,indent=4)
            print(f'train_data:{len(data)} eval_num:{len(eval_data)} dev_ratio:{dev_ratio}')
        self.data_dict = {'dia':data}

        self.datacollatorforseq2seq = transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)
        self.ignore_index = -100
        self.debug = True
        
        self.lengths = {k: len(self.data_dict[k]) for k in self.data_dict.keys()}
        self.keys = list(self.data_dict.keys())
        self.epoch = 1
        self.data_priority = {k:1 for k in self.keys}
        self.data_epoch =  {k: self.epoch for k in self.keys}

        self.epoch_id = []
        self.weights = []
        self.pos_key = []
        for keyi,key in enumerate(self.keys):
            priority = self.data_priority[key]
            epoch = self.data_epoch[key]
            decay = 0.1
            for id in range(epoch):
                self.weights += [priority] * int(self.lengths[key])
                self.pos_key += [keyi] * int(self.lengths[key])
                self.epoch_id += [id] * int(self.lengths[key])
                priority = priority * decay

    def __getitem__(self, index):
        key = self.keys[self.pos_key[index]]
        sub_index = index % self.lengths[key]
        da = self.preprocess(self.data_dict[key][sub_index])
        # da['data_type'] = key
        da['data_type'] = str(self.epoch_id[index])
        return da

    def get_data_info(self):
        res = {}
        total = 0
        for k,v in self.data_epoch.items():
            res[k] = self.lengths[k]*v
            total += self.lengths[k]*v
        res['sum'] = total
        return res

    def get_sym(self,raw_sym, current_syms = [],is_en = False):
        true_t = '"{}"'
        if not is_en:
            false_t = '没有"{}"'
        else:
            false_t = 'No "{}"'
        syms = [true_t.format(s) if s[:3] != 'No ' else false_t.format(s[3:]) for s in raw_sym]
        return [s for s in syms if s not in current_syms]

    def get_candidate_dis(self,candidate_disease,is_en = False):
        max_len = 60
        pre_str = '- "{}"'
        if is_en:
            post_str = ['.\n',', typically characterized by {}.\n']
        else:
            post_str = ['。\n','，该病常见的症状为{}。\n']
        dis_str = ''
        for dis in candidate_disease:
            if dis['common_symptom'] is not None:
                dis_str += pre_str.format(dis['disease']) + post_str[1].format(dis['common_symptom'][:max_len])
            else:
                dis_str += pre_str.format(dis['disease']) + post_str[0]
        
        return dis_str.strip()
    
    def get_inference_str(self,diagnosis):
        temp = '{{"{}": {}, "{}": {}, "{}": {} }}'
        tmp = []
        for k,v in diagnosis.items():
            tmp.extend([k,str(v)])
        return temp.format(*tmp)

    def preprocess(self, data):
        input_ids = []
        labels = []
        # Determine whether it is normal chat or diagnostic chat
        if isinstance(data,dict) and "target_disease" in data:
            conv = data['conversations']
            sep = '\n'
            sep_ids = self.tokenizer.encode(sep,add_special_tokens= False)
            assert len(sep_ids) == 1
            sep_ids = sep_ids[0]
            round_num = len(conv) // 2
            sym_info = []
            is_en = 'en_' in data['case_id']
            if is_en:
                t1 = ["## Enter the diagnostic process, analyzing patient symptoms:\n{}","## Analyzing patient symptoms:\n{}"]
                t2 = '\n## Based on the information provided, the likely diagnoses include:\n{}\n\n## Diagnostic reasoning:\n'
                t3 = '{}\n\n## Diagnostic confidence:\n{}'
                t4 = ['\n\n## Inadequate for diagnosis. Ask for symptoms:\n{}','\n\n## Diagnosis:\n{}']
            else:
                t1 = ["## 进入诊断流程，分析病人症状信息:\n{}","## 总结病人症状信息:\n{}"]
                t2 = '\n## 根据现有信息，病人可能患有的疾病为:\n{}\n\n## 诊断推理:\n'
                t3 = '{}\n\n## 诊断置信度:\n{}'
                t4 = ['\n\n## 为了更好的诊断疾病，我还需要了解您更多信息，请您回答我的问题:\n{}','\n\n## 做出诊断:\n{}']

            for ind in range(round_num):

                h = conv[ind*2]['value'].strip()
                h = f"<|user|>\n{h}\n"
                cur_input_ids = self.tokenizer(h, add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids
                input_ids += cur_input_ids
                labels += [self.ignore_index] * len(cur_input_ids)

                diadata = conv[ind*2+1]

                # t1 
                if len(sym_info) == 0:
                    template = t1[0]
                else:
                    template = t1[1]
                sym_info.extend(self.get_sym(diadata['symptom_abstraction'], sym_info, is_en=is_en))
                symptoms = template.format(', '.join(sym_info))
                g = f"<|assistant|>\n{symptoms}"
                
                cur_input_ids = self.tokenizer(g, add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids
                input_ids += cur_input_ids
                labels += [self.ignore_index] + cur_input_ids[1:]
                input_ids.append(sep_ids)
                labels.append(self.tokenizer.eos_token_id)

                # t2
                candi = t2.format(self.get_candidate_dis(diadata['disease_recall'],is_en=is_en))
                cur_input_ids = self.tokenizer(candi, add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids
                input_ids += cur_input_ids
                labels += [self.ignore_index] * len(cur_input_ids)
                
                # t3
                diagnosis = t3.format(diadata['diagnostic_reasoning'] ,self.get_inference_str(diadata['confidence_assessment']))
                cur_input_ids = self.tokenizer(diagnosis, add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids
                input_ids += cur_input_ids
                labels += cur_input_ids
                input_ids.append(sep_ids)
                labels.append(self.tokenizer.eos_token_id)

                if len(diadata['value']) == 0:
                    # Last turn
                    assert ind == round_num - 1
                    break

                # t4 
                if diadata['decision'] == 'diagnosis':
                    assert ind == round_num - 1
                    # Stop
                    template = t4[1]
                    conclusion = template.format(diadata['value'])
                    cur_input_ids = self.tokenizer(conclusion, add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids
                    input_ids += cur_input_ids
                    ignore_len = len(self.tokenizer(template.format(''), add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids)
                    labels += [self.ignore_index] * ignore_len + cur_input_ids[ignore_len:]
                    input_ids.append(self.tokenizer.eos_token_id)
                    labels.append(self.tokenizer.eos_token_id)
                    break
                else:
                    template = t4[0]
                    conclusion = template.format(diadata['value'])
                    cur_input_ids = self.tokenizer(conclusion, add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids
                    input_ids += cur_input_ids
                    ignore_len = len(self.tokenizer(template.format(''), add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids)
                    labels += [self.ignore_index] * ignore_len + cur_input_ids[ignore_len:]
                    input_ids.append(sep_ids)
                    labels.append(self.tokenizer.eos_token_id)
            
            # 确保最后eos
            input_ids[-1] = self.tokenizer.eos_token_id

        else:
            # 非诊断数据，普通对话数据
            conv = data['conversations']
            data = []
            for da in conv:
                data.append(da['value'])

            sep = '\n'
            sep_ids = self.tokenizer.encode(sep,add_special_tokens= False)
            assert len(sep_ids) == 1
            sep_ids = sep_ids[0]
            round_num = len(data) // 2

            for ind in range(round_num):
                
                h = data[ind*2].strip()
                h = f"<|user|>\n{h}\n" # [1, 29871, 518, 25580, 29962, 29871, human, 518, 29914, 25580, 29962, 29871]

                g = data[ind*2+1]
                g = f"<|assistant|>\n{g}"
                
                cur_h_input_ids = self.tokenizer(h, add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids
                cur_g_input_ids = self.tokenizer(g, add_special_tokens= False, max_length=self.config.max_seq_len, truncation=True).input_ids

                if len(cur_g_input_ids) + len(cur_h_input_ids) + len(input_ids) >= self.config.max_seq_len and len(input_ids) > 0 :
                    input_ids = input_ids[:-1]
                    labels = labels[:-1]
                    break
                
                input_ids += cur_h_input_ids
                labels += [self.ignore_index] * len(cur_h_input_ids) # check this

                input_ids += cur_g_input_ids
                labels += [self.ignore_index] + cur_g_input_ids[1:]

                if ind < round_num - 1:
                    input_ids.append(sep_ids)
                    labels.append(self.tokenizer.eos_token_id)
                    # prompt_str += sep 
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

        if self.debug:
            print('input_ids',self.tokenizer.decode(input_ids))
            labels = [item if item != self.ignore_index else self.tokenizer.pad_token_id for item in labels]
            print('labels',self.tokenizer.convert_ids_to_tokens(labels))
            self.debug = False
        return {'input_ids': input_ids[:self.config.max_seq_len], 'labels': labels[:self.config.max_seq_len]}

    def __len__(self):
        return len(self.weights)

    def sample_num(self):
        # return sum([int(len(self.data_dict[daname])*daweight) for daname, daweight in self.file_weight.items()])
        return len(self.weights)

    def collate_fn(self, batch):
        # processed_batch = [self.preprocess(x) for x in batch]
        # return self.datacollatorforseq2seq(processed_batch)
        # pass
        return batch


parser = argparse.ArgumentParser(description='Args of sft')
# Model Args
parser.add_argument('--data_dir', default='./CoD_en.json', type=str)
parser.add_argument('--model_path', default='Yi-34B', type=str)
args = parser.parse_args()
args.num_workers = 12
args.train_bsz_per_gpu = 256
args.max_seq_len = 4096
args.save_path = '.'.join(os.path.split(args.data_dir)[-1].split('.')[:-1])+'_'+os.path.split(args.model_path)[-1]+f'_{args.max_seq_len}_dataset'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
args.experiment_name = 'huatuo2_datapre'

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = '<PAD>'

#%%
train_dataset = HuatuoGPT_data(args, tokenizer)
# train_dataset = HuatuoGPT_data(args, tokenizer, 0.03)
sampler = WeightedRandomSampler(train_dataset.weights, num_samples=train_dataset.sample_num(), replacement=False)
train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, sampler=sampler, drop_last=False, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers)

train_dataloader_iterator = tqdm(enumerate(train_dataloader))
args.log_step = len(train_dataloader) // 30
if args.log_step == 0:
    args.log_step  = 100

from collections import defaultdict
key_nums = defaultdict(int)

all_inputs_ids = []
all_labels = []
pad_id = tokenizer.pad_token_id
ignore_index = -100
for batch_cnt, batch in train_dataloader_iterator:
    cur_input = []
    cur_label = []
    for da in batch:
        key_nums[da['data_type']] += 1
        if len(da['input_ids']) + len(cur_input) <= args.max_seq_len:
            cur_input += da['input_ids']
            cur_label +=  da['labels']
        else:
            pad_len = args.max_seq_len - len(cur_input)
            cur_input += [pad_id] * pad_len
            cur_label += [ignore_index] * pad_len
            assert len(cur_input) == len(cur_label) == args.max_seq_len, f'{len(cur_input)} {len(cur_label)}'
            all_inputs_ids.append(cur_input)
            all_labels.append(cur_label)
            cur_input = da['input_ids']
            cur_label =  da['labels']
    pad_len = args.max_seq_len - len(cur_input)
    cur_input += [pad_id] * pad_len
    cur_label += [ignore_index] * pad_len
    all_inputs_ids.append(cur_input)
    all_labels.append(cur_label)
    assert len(cur_input) == len(cur_label) == args.max_seq_len, f'{len(cur_input)},{len(cur_label)}'

    if batch_cnt % args.log_step == 0:
        logdata = {}
        for key in key_nums:
            logdata[key + '_num'] = key_nums[key]
        key_nums = defaultdict(int)

assert len(all_inputs_ids) == len(all_labels)
print(len(all_inputs_ids))
save_dataset = datasets.Dataset.from_dict({'input_ids': all_inputs_ids, 'labels':all_labels})
save_dataset.save_to_disk(args.save_path)

print(json.dumps(train_dataset.data_priority,ensure_ascii=False,indent=2),json.dumps(train_dataset.data_epoch,ensure_ascii=False,indent=2),json.dumps(train_dataset.get_data_info(),ensure_ascii=False,indent=2))
