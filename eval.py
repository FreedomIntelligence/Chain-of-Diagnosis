"""
Test DxBench Muzhi Dxy
"""

import os
import copy
import json
import torch
import logging
import argparse

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import  DataLoader
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup
import json
from copy import deepcopy
import random
import requests
from retrying import retry
import re

from cod_cli import DiagnosisChatbot


def get_max_kv(inference):
    max_key = max(inference, key=inference.get)
    max_value = inference[max_key]
    return max_key,max_value


def scorer(data):
    acc_dict = {}
    global wrong_analysis
    wrong_analysis = []
    true_analysis = []
    wrong_num = 0
    acc_res = [[0,0],[0,0],[0,0]]

    for da in data:
        if len(da['chat_history']) == 0:
            wrong_num += 1
            continue

        try:
            pattern = r"## (?:Diagnostic confidence:|诊断置信度:)\s*({.*})"
            match = re.search(pattern, da['chat_history'][-1][-1] , re.DOTALL)
            confidence_distribution_last_turn = json.loads(match.group(1))
            predict_last_turn, value_last_turn =  get_max_kv(confidence_distribution_last_turn)
            
            match = re.search(pattern, da['chat_history'][0][-1] , re.DOTALL)
            confidence_distribution_wo_inquiry = json.loads(match.group(1))
            predict_wo_inquiry, value_wo_inquiry =  get_max_kv(confidence_distribution_wo_inquiry)

        except Exception as e:
            print('Wrong')
            print(e)
            print(match)
            wrong_num += 1
            continue
        
        acc_res[1][0] += 1
        acc_res[1][1] += len(da['chat_history'])


        acc_res[0][0] += 1
        if da['disease'] == predict_last_turn:
            acc_res[0][1] += 1
            true_analysis.append([da['chat_history'][-1][-1], da['disease']])
        else:
            wrong_analysis.append([da['chat_history'][-1][-1], da['disease']])

        
        if da['disease'] == predict_wo_inquiry:
            acc_res[2][1] += 1
            true_analysis.append([da['chat_history'][-1][-1], da['disease']])
        else:
            wrong_analysis.append([da['chat_history'][-1][-1], da['disease']])

    res = {}

    res[f'ACC_wo_inquiry'] = acc_res[2][1] / acc_res[0][0]
    res[f'ACC'] = acc_res[0][1] / acc_res[0][0]
    res[f'ask_turn'] = acc_res[1][1] / acc_res[1][0]
    res[f'len'] = acc_res[0][0]
    res['wrong_num'] = wrong_num
    return res


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path):
        self.config = config

        with open(data_path) as f:
            self.dataset = json.load(f)
        self.datas = []
        for da in self.dataset:
            if len(da['explicit_symptoms']) > 0:
                self.datas.append(da)


    def __getitem__(self, index):
        da = self.datas[index]
        return {
            'data': da
        }
    
    def __len__(self):
        return len(self.datas)
    
    def collate_fn(self, batch):
        batch_data = [x['data'] for x in batch]
        out_batch = {}
        out_batch['data'] = batch_data
        return out_batch

def table_to_csv_string(table):
    rows = [",".join(table.columns)]  # 添加标题行
    for row in table.data:
        rows.append(",".join(map(str, row)))
    return "\n".join(rows)


class GPT:
    def __init__(self,model_name = 'gpt-4-turbo') -> None:
        self.key_ind = 0
        self.init_api_keys()
        self.max_wrong_time = 5
        self.model_name = model_name
        print(f'use model of {self.model_name}')

    def init_api_keys(self):
        self.keys = ['your api key !!']
        self.wrong_time = [0]*len(self.keys)
        random.shuffle(self.keys)
    
    def get_api_key(self):
        self.key_ind =  (self.key_ind + 1) % len(self.keys)
        return self.keys[self.key_ind]

    def call(self, content, args = {}, showkeys = False):
        api_key = self.get_api_key()
        if showkeys:
            print(api_key)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # "OpenAI-Organization": organization,
        }
        if isinstance(content,str):
            parameters = {
                "model": self.model_name,
                "messages": [{'role': 'user', 'content': content}],
                **args,
            }
        else:
            parameters = {
                "model": self.model_name,
                "messages": content,
                **args,
            }
        response = requests.post(
            url,
            headers=headers,
            json=parameters,
            timeout = 5
            # verify=False
        )
        response = json.loads(response.content.decode("utf-8"))
        if 'error' in response:
            self.wrong_time[self.key_ind] += 1
            if self.wrong_time[self.key_ind] > self.max_wrong_time:
                print(response)
                print(f'del {self.keys[self.key_ind]}')
            assert False, str(response)
        return response['choices'][0]['message']['content']
    
    @retry(wait_fixed=1000, stop_max_attempt_number=20)
    def retry_call(self, content, args = {}):
        return self.call(content, args)

# Chinese Prompt
query_prompt = """你是一名患者，下面是你的症状信息：
{}

你的实际疾病是 "{}"

你需要回答医生的问题:
"{}"

请根据你的症状信息和疾病，直接回答医生问题，只回答 "是" 或 "不是" 就好，不要输出其他内容。"""

# English Prompt
# query_prompt = """You are a patient, here are your symptom details:
# {}

# Your actual disease is {}.

# You need to answer the doctor’s question:
# {}

# Please answer the doctor’s question based on your symptom information and disease, simply reply with "yes" or "no", and do not include any other content."""



def test(args):
    accelerator = Accelerator()
    torch.cuda.set_device(accelerator.process_index)
    accelerator.print(f'args:\n{args}')

    chatbot = DiagnosisChatbot(args.model_path,confidence_threshold=args.threshold)

    accelerator.print(f'load_finish')

    dataset = TestDataset(args, args.data_path)

    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)
    val_dataloader = accelerator.prepare( val_dataloader)

    accelerator.wait_for_everyone()
    cache_data = []
    gpt = GPT(model_name='gpt-4-turbo')

    with torch.no_grad():
        ress = []
        
        dataloader_iterator = tqdm(val_dataloader, total=len(val_dataloader)) if accelerator.is_main_process else val_dataloader

        max_inquiry = args.max_inquiry
        threshold = args.threshold
        try_time = 2
        isdebug = True
        for batch in dataloader_iterator:
            data = batch["data"]

            for da in data:
                gpt4_requests = []
                history = []

                # for chinese
                exp_request =  ', '.join([ k if v else f'没有"{k}"' for k,v in da['explicit_symptoms'] ])
                all_sym_info = ', '.join([ k if v else f'没有"{k}"' for k,v in da['explicit_symptoms'] ] + [ k if v else f'没有"{k}"' for k,v in da['implicit_symptoms'] ])

                # for english
                # exp_request =  ', '.join([ k if v else f'No "{k}"' for k,v in da['explicit_symptoms'] ])
                # all_sym_info = ', '.join([ k if v else f'No "{k}"' for k,v in da['explicit_symptoms'] ] + [ k if v else f'No "{k}"' for k,v in da['implicit_symptoms'] ])

                query = exp_request
                chatbot.history = []
                for jj in range(max_inquiry+1):
                    # start converastion
                    for ii in range(try_time):
                        cur_generate, newhistory, confidence_distribution  = chatbot.inference(query, history, da['candidate_diseases'])
                        if len(confidence_distribution) > 0:
                            break

                    if 'Diagnosis:' in cur_generate or '做出诊断:' in cur_generate:
                        break

                    if jj == max_inquiry:
                        # Last turn
                        break
                    
                    if 'Ask for symptoms:\n' in cur_generate:
                        sub_string = 'Ask for symptoms:\n'
                    elif '请您回答我的问题:\n' in cur_generate:
                        sub_string = '请您回答我的问题:\n'
                    else:
                        print('Wrong',cur_generate)
                        break

                    start_index = cur_generate.find(sub_string)
                    start_index += len(sub_string)
                    doc_query = cur_generate[start_index:].strip()

                    if jj != max_inquiry-1:
                        gpt4query = query_prompt.format(all_sym_info,da['disease'],doc_query)
                        gpt4_requests.append(gpt4query)
                        query = gpt.retry_call(gpt4query)
                        history = newhistory
                        if isdebug:
                            isdebug = False
                            print(f'gpt4 query {gpt4query}\nresponse{query}\n',flush=True)

                da['sym_info'] = copy.deepcopy(chatbot.sym_info)
                da['chat_history'] = newhistory
                da['final_confidence_distribution'] = confidence_distribution
                da['gpt4_requests'] = gpt4_requests
            cache_data.extend(data)

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        
        all_data =  [None] * dist.get_world_size()

        dist.all_gather_object(all_data,cache_data)

        all_data = [item for sublist in all_data for item in sublist]

        ress.extend(all_data)

        if accelerator.is_main_process:
            task_name = os.path.split(args.model_path)[-1].split(".")[0] if 'tfmr' not in args.model_path else '-'.join(args.model_path.split('/')[-3:-1])
            task_name =  task_name + f'_{args.threshold}_{args.max_inquiry}_{os.path.split(args.data_path)[-1].replace(".json","")}'
            run_time = 0
            for i in range(100):
                run_time += 1
                out_file = f'result/output/{task_name}_t{run_time}.json'
                if not os.path.exists(out_file):
                    break
            with open(out_file, 'w', encoding='utf-8') as fw:
                json.dump(ress,fw,ensure_ascii=False,indent=4)

            print(f'test results: {out_file}')
            val_res = scorer(ress)
            outstr = json.dumps(val_res,ensure_ascii=False,indent = 2)
            
            accelerator.print(outstr)
            with open(f'result/{task_name}_t{run_time}.json','w', encoding='utf-8') as f:
                f.write(outstr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')

    # Model Args
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--max_inquiry', default=0, type=int)
    parser.add_argument('--threshold', default= 0.5, type=float)

    # Other Args
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    set_seed(args.seed)
    test(args)      