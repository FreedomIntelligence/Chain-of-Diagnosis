# 支持英文
import os
import platform
import torch
from threading import Thread
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import argparse
from transformers import TextIteratorStreamer
from transformers.generation.utils import GenerationConfig
import json
import re

import numpy as np
from retrieve_utils import (
    construct_flatindex_from_embeddings, 
     convert_index_to_gpu,
     encoder
)
import json
from transformers import LlamaForCausalLM


class DiagnosisChatbot():
    def __init__(self, model_dir,confidence_threshold = 0.5):

        self.max_len = 4096
        self.gen_kwargs = {'max_new_tokens': 768, 'do_sample':True, 'top_p':0.7, 'temperature':0.5, 'repetition_penalty':1.1}

        model, tokenizer = self.load_model(model_dir)
        self.model = model
        self.tokenizer = tokenizer
        self.sep = tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)
        
        excluded_ids = ['1656164150671335426', '1656164150562283522', '1656164170883686402', '1656164187891589122', '1656164159110275073', '1656164142131732482', '1656164167129784322', '1656164140797943810', '1656164130668699650', '1656164130798723073', '1656164140768583682', '1656164167251419137', '1656164134657482753', '1656164159026388994', '1656164140818915330', '1656164181088428034', '1656164153162752001', '1656164133688598529', '1656164142135926785', '1656164164525121538', '1656164185387589633']
        excluded_ids_en = ['1656164174583062529', '1656164139271217154', '1656164172892758018', '1656164182577405953', '1656164182594183170', '1656164134418407426', '1656164172947283969', '1656164131574669313', '1656164190991179777', '1656164183869251585', '1656164151631831041', '1656164134384852994', '1656164160184016898', '1656164190873739265', '1656164184800387073', '1656164174494982146', '1656164147903094785', '1656164160087547906', '1656164168585207809', '1656164134472933377', '1656164183189774338', '1656164174901829635', '1656164152919482370', '1656164179532341249', '1656164141842325505', '1656164182682263553', '1656164151900266497', '1656164192618569730', '1656164136515559426', '1656164161626857473', '1656164154643341314', '1656164188592037890', '1656164156337840130', '1656164193751031810', '1656164185794437122', '1656164179209379842', '1656164164965523457', '1656164179310043137', '1656164179209379843', '1656164171294728193', '1656164188600426498', '1656164171584135170', '1656164184485814273', '1656164142261755906', '1656164193323212801', '1656164162025316354', '1656164152219033602', '1656164134061891586', '1656164193365155842', '1656164190978596865', '1656164179452649474', '1656164193918803971', '1656164136456839169', '1656164169449234433', '1656164192643735555', '1656164136893046785', '1656164193532928002', '1656164167213670402', '1656164171970011137', '1656164157814235138', '1656164152344862722', '1656164141129293825', '1656164160284680194', '1656164182636126210', '1656164174855692290', '1656164185802825730', '1656164165061992450', '1656164131801161730', '1656164152953036802', '1656164186138370050', '1656164185303703553', '1656164182648709122', '1656164183927971841', '1656164182757761025', '1656164152508440577', '1656164190856962050', '1656164145302626305', '1656164182585794561', '1656164192840867841', '1656164134812672001', '1656164171751907329', '1656164189644808194', '1656164171269562370', '1656164142249172994', '1656164190861156355', '1656164172783706114', '1656164171575746562', '1656164190848573443', '1656164167113007106', '1656164152336474115', '1656164134355492865', '1656164157537411074', '1656164185391783938', '1656164134468739074', '1656164151065600002', '1656164189485424641', '1656164168648122370', '1656164153263415298', '1656164185060433922', '1656164141318037506', '1656164157529022466', '1656164189607059457', '1656164142203035650', '1656164190609498113', '1656164193566482433', '1656164154332962818', '1656164144233078785', '1656164131155238913', '1656164190299119618', '1656164178823503873', '1656164167037509633', '1656164190525612033', '1656164185752494081', '1656164182912950274', '1656164144144998401', '1656164192903782402', '1656164193394515970', '1656164184993325058', '1656164157331890177', '1656164162281168897', '1656164184653586433', '1656164192941531138', '1656164145252294657', '1656164141737467906', '1656164164621590531', '1656164137140510721', '1656164144367296514', '1656164183542095874', '1656164141880074242', '1656164152336474114', '1656164184628420609', '1656164179322626049', '1656164169415680001', '1656164161886904321', '1656164142194647041', '1656164169168216066', '1656164164910997506', '1656164184594866178', '1656164141473226753', '1656164145562673154', '1656164157600325633', '1656164174553702401', '1656164142320476162', '1656164159177383938', '1656164128898703361', '1656164187950309377', '1656164152948842497', '1656164131662749697', '1656164192949919745', '1656164157621297153', '1656164159307407362', '1656164165162655746', '1656164173245079553', '1656164147211034626', '1656164171730935809', '1656164152382611458', '1656164174557896705', '1656164145407483907', '1656164144480542721', '1656164173056335874', '1656164147085205505', '1656164134498099202', '1656164159923970050', '1656164144988053505', '1656164184095744001', '1656164141523558402', '1656164145512341505', '1656164141938794498', '1656164171827404801', '1656164168790728706', '1656164151409532930', '1656164174746640385', '1656164152328085506', '1656164178722840577', '1656164162285363201', '1656164187681873922', '1656164141636804610', '1656164147013902337', '1656164174906023937', '1656164175166070786', '1656164192996057090', '1656164172720791553', '1656164193377738753', '1656164166605496321', '1656164174952161282', '1656164131604029442', '1656164142043652098', '1656164184045412354', '1656164190907293698', '1656164159928164353', '1656164141225762817', '1656164162146951169', '1656164168664899585', '1656164161597497346', '1656164141963960321', '1656164190311702529', '1656164193755226113', '1656164173144416257', '1656164190613692417', '1656164183546290178', '1656164185798631425', '1656164172674654209', '1656164141485809665', '1656164147286532098', '1656164182598377474', '1656164136477810689', '1656164141078962177', '1656164157193478145', '1656164164650950657', '1656164171688992770', '1656164172817260546', '1656164185387589634', '1656164159353544707', '1656164174549508098', '1656164182581600257', '1656164182669680641', '1656164174545313793', '1656164134326132738', '1656164188684312579', '1656164134405824514', '1656164151619248129', '1656164172695625730', '1656164141217374209', '1656164152512634882', '1656164141716496386', '1656164193289658369', '1656164137379586050', '1656164164634173442', '1656164169411485697', '1656164134460350465', '1656164172683042818', '1656164134842032130', '1656164174956355586', '1656164178815115266', '1656164174570479617', '1656164142312087553', '1656164157793263617', '1656164152713961473', '1656164171839987713', '1656164172724985857', '1656164188629786626', '1656164193574871041', '1656164142232395777', '1656164134447767554', '1656164192887005186', '1656164174566285314', '1656164157134757890', '1656164131708887042', '1656164172049702913', '1656164171370225666', '1656164193973329922', '1656164185844768770', '1656164156711133186', '1656164193893638146', '1656164179410706434', '1656164133973811201', '1656164172406218754', '1656164172867592193', '1656164152206450691', '1656164152806236161', '1656164157113786369', '1656164178638954498', '1656164185781854210', '1656164134527459330', '1656164145323597826', '1656164147886317570', '1656164171445723138', '1656164144216301569', '1656164188508151811', '1656164179205185537', '1656164144887390210', '1656164185890906114', '1656164182573211650', '1656164175010881538', '1656164165162655747', '1656164141846519809', '1656164154408460289', '1656164182938116099', '1656164132103151617', '1656164144568623105', '1656164190911488002', '1656164182464159745', '1656164164642562050', '1656164142198841345', '1656164182774538241', '1656164172108423170', '1656164167196893185', '1656164131658555393', '1656164172435578882', '1656164179528146946', '1656164159999467523', '1656164161844961283', '1656164167205281794', '1656164145378123778', '1656164171764490242', '1656164188801753090', '1656164174994104322', '1656164190294925314', '1656164165154267138', '1656164153389244419', '1656164189275709442', '1656164172615933953', '1656164185446309890', '1656164170602668033', '1656164157478690817', '1656164192182362113', '1656164147517218819', '1656164147815014403', '1656164179498786817', '1656164193348378625', '1656164134259023873', '1656164168681676802', '1656164134464544770', '1656164157411581953', '1656164169243713537', '1656164171705769985']

        # Chinese Retriever
        retriever_zh =  self.load_retrieval(os.path.join(model_dir,'retriever/zh/encoder') ,
                    os.path.join(model_dir,'retriever/zh/index') , 
                    os.path.join(model_dir,'retriever/zh/disease_database_zh.json'),excluded_disease_id=excluded_ids)
        
        # English Retriever
        retriever_en =  self.load_retrieval(os.path.join(model_dir,'retriever/en/encoder') ,
                    os.path.join(model_dir,'retriever/en/index') , 
                    os.path.join(model_dir,'retriever/en/disease_database_en.json'),excluded_disease_id=excluded_ids+excluded_ids_en)

        self.retriever_zh = retriever_zh
        self.retriever_en = retriever_en
        self.confidence_threshold = confidence_threshold
        self.min_sym_num = 2
        self.history = []
        self.sym_info = {'true_syms':[],'false_syms':[]}

    def load_retrieval(self, model_path,pembed_dir,data_path,excluded_disease_id = []):
        model = encoder(model_path)
        model.to('cuda')
        model.eval()
        passage_embeddings = np.memmap(os.path.join(pembed_dir, "passages.memmap"), dtype=np.float32, mode="r"
            ).reshape(-1, model.output_embedding_size)
        index = construct_flatindex_from_embeddings(passage_embeddings)
        # 加载到GPU
        # index = convert_index_to_gpu(index, faiss_gpu_index = 0, useFloat16 = False)
        disease = []
        diseaseid2id = {}
        disease2id = {}
        with open(data_path) as f:
            datas = json.load(f)
            for id,da in enumerate(datas):
                disease.append(da)
                diseaseid2id[da['disease_id']] = id
                disease2id[da['disease']] = id
        model.index = index
        model.disease = disease
        model.diseaseid2id = diseaseid2id
        model.disease2id = disease2id
        exclude_ids = []
        for dis_id in excluded_disease_id:
            exclude_ids.append(diseaseid2id[dis_id])
        model.exclude_ids = exclude_ids
        return model

    def load_model(self,model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir,  torch_dtype=torch.bfloat16).cuda()
        model = model.eval()
        return model, tokenizer

    def generate_prompt(self,query, history):
        if not history:
            return  f"<|user|>\n{query}\n<|assistant|>\n"
        else:
            prompt = ''
            for i, (old_query, response) in enumerate(history):
                prompt += "<|user|>\n{}\n<|assistant|>\n{}\n".format(old_query, response)
            prompt += "<|user|>\n{}\n<|assistant|>\n".format(query)
            return prompt

    def remove_overlap(self,str1, str2):
        for i in range(len(str1), -1, -1): 
            if str1.endswith(str2[:i]): 
                return str2[i:] 
        return str2 

    def get_candidate_dis(self,candidate_disease,is_en = False):
        max_len = 60
        pre_str = '- "{}"'
        if is_en:
            t2 = '\n\n##  Based on the information provided, the likely diagnoses include:\n{}\n\n## Diagnostic reasoning:\n'
            post_str = ['.\n',', typically characterized by {}.\n']
        else:
            t2 = '\n\n## 根据现有信息，病人可能患有的疾病为:\n{}\n\n## 诊断推理:\n'
            post_str = ['。\n','，该病常见的症状为{}。\n']
        dis_str = ''
        for dis in candidate_disease:
            if dis['common_symptom'] is not None:
                dis_str += pre_str.format(dis['disease']) + post_str[1].format(dis['common_symptom'][:max_len])
            else:
                dis_str += pre_str.format(dis['disease']) + post_str[0]
        return t2.format(dis_str.rstrip())


    @torch.no_grad()
    def model_genrate_streaming(self, prompt):
        inputs = self.tokenizer([prompt], add_special_tokens= False, return_tensors="pt").input_ids
        inputs = inputs[:, -self.max_len:]
        inputs = inputs.to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer,skip_prompt=True)
        generation_kwargs = dict(input_ids=inputs, streamer=streamer, **self.gen_kwargs)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ''

        for new_text in streamer:
            if self.sep in new_text:
                new_text = self.remove_overlap(generated_text,new_text[:-len(self.sep)])
                for char in new_text:
                    generated_text += char
                    yield char
                break
            for char in new_text:
                generated_text += char
                yield char

    @torch.no_grad()
    def model_genrate(self, prompt):
        inputs = self.tokenizer([prompt], add_special_tokens= False, return_tensors="pt").input_ids
        inputs = inputs[:, -self.max_len:]
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(inputs, **self.gen_kwargs)
        res = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return res

    def get_sym_num(self):
        return len(self.sym_info['true_syms']) + len(self.sym_info['false_syms'])

    def chat(self, query):
        prompt = self.generate_prompt(query, self.history)
        # print(prompt,flush=True)
        generated_text = ''
        for char in self.model_genrate_streaming(prompt):
            generated_text += char
            yield char
        cur_generate = generated_text

        # For diagnosis.
        if  'Enter the diagnostic process, analyzing patient symptoms:' in generated_text or 'Analyzing patient symptoms:' in generated_text or '进入诊断流程，分析病人症状信息:' in generated_text or '总结病人症状信息:' in generated_text:
            if 'Enter the diagnostic process, analyzing patient symptoms:' in generated_text or 'Analyzing patient symptoms:' in generated_text:
                is_en = True
            else:
                is_en = False

            # Extracting symptoms
            p_syms = re.findall(r'(?:\n|,)\s*"([^"]*?)\s*"', generated_text)
            f_syms = re.findall(r'(?:没有|No)\s*"([^"]*?)\s*"', generated_text)

            self.sym_info = {'true_syms':[],'false_syms':[]}
            for sym in p_syms:
                if sym not in self.sym_info['true_syms']:
                    self.sym_info['true_syms'].append(sym)
            for sym in f_syms:
                if sym not in self.sym_info['false_syms']:
                    self.sym_info['false_syms'].append(sym)
            # print(self.sym_info,flush=True)
            if is_en:
                candi = self.retriever_en.find_top_k(self.sym_info['true_syms'], self.sym_info['false_syms'])
            else:
                candi = self.retriever_zh.find_top_k(self.sym_info['true_syms'], self.sym_info['false_syms'])

            t2_str =  self.get_candidate_dis(candi,is_en)

            # print(t2_str,end='',flush = True)
            yield t2_str

            cur_generate += t2_str
            generated_text = ''
            for char in self.model_genrate_streaming(prompt+cur_generate):
                generated_text += char
                yield char
            # print('\n',end='',flush=True)
            # generated_text += '\n'

            try:
                if is_en:
                    pattern = r"## Diagnostic confidence:\s*(.*)"
                else:
                    pattern = r"## 诊断置信度:\s*(.*)"
                match = re.search(pattern, generated_text, re.DOTALL)
                confidence_distribution = json.loads(match.group(1))
            except Exception as e:
                print('Error')
                print(e)
                print(confidence_distribution)
                return cur_generate + generated_text

            cur_generate += generated_text
            max_key = max(confidence_distribution, key=confidence_distribution.get)
            max_value = confidence_distribution[max_key]
            
            # if max_value > self.confidence_threshold and (self.get_sym_num(self.sym_info) >= self.min_sym_num or max_value > self.confidence_threshold +0.2 ):
            if max_value > self.confidence_threshold:
                # 诊断
                if is_en:
                    t4_1 = '\n\n## Diagnosis:\n'
                else:
                    t4_1 = '\n\n## 做出诊断:\n'
                # print(t4_1,end='',flush = True)
                yield t4_1
                cur_generate += t4_1
                generated_text = ''
                for char in self.model_genrate_streaming(prompt+cur_generate):
                    generated_text += char
                    yield char
                cur_generate += generated_text
                
                self.history = self.history + [(query, cur_generate)]
            else:
                # Diagnosis
                if is_en:
                    t4_2 = '\n\n\nInadequate for diagnosis. Ask for symptoms:\n'
                else:
                    t4_2 = '\n\n\n为了更好的诊断疾病，我还需要了解您更多信息，请您回答我的问题:\n'
                # print(t4_2,end='',flush = True)
                yield t4_2
                cur_generate += t4_2
                generated_text = ''
                for char in self.model_genrate_streaming(prompt+cur_generate):
                    generated_text += char
                    yield char
                cur_generate += generated_text

                self.history = self.history + [(query, cur_generate)]
        else:
            self.history = self.history + [(query, cur_generate)]
        
    def inference(self, query, history = [], candidate_diseases = None):
        self.history = history
        prompt = self.generate_prompt(query, self.history)
        # print(prompt,flush=True)
        generated_text = self.model_genrate(prompt)
        cur_generate = generated_text

        # For diagnosis.
        if  'Enter the diagnostic process, analyzing patient symptoms:' in generated_text or 'Analyzing patient symptoms:' in generated_text or '进入诊断流程，分析病人症状信息:' in generated_text or '总结病人症状信息:' in generated_text:
            if 'Enter the diagnostic process, analyzing patient symptoms:' in generated_text or 'Analyzing patient symptoms:' in generated_text:
                is_en = True
            else:
                is_en = False

            p_syms = re.findall(r'(?:\n|,)\s*"([^"]*?)\s*"', generated_text)
            f_syms = re.findall(r'(?:没有|No)\s*"([^"]*?)\s*"', generated_text)

            self.sym_info = {'true_syms':[],'false_syms':[]}
            for sym in p_syms:
                if sym not in self.sym_info['true_syms']:
                    self.sym_info['true_syms'].append(sym)
            for sym in f_syms:
                if sym not in self.sym_info['false_syms']:
                    self.sym_info['false_syms'].append(sym)

            if is_en:
                if not candidate_diseases:
                    candi = self.retriever_en.find_top_k(self.sym_info['true_syms'], self.sym_info['false_syms'])
                else:
                    candi = self.retriever_en.find_top_k_with_candis(self.sym_info['true_syms'], self.sym_info['false_syms'],candidate_diseases)
            else:
                if not candidate_diseases:
                    candi = self.retriever_zh.find_top_k(self.sym_info['true_syms'], self.sym_info['false_syms'])
                else:
                    candi = self.retriever_zh.find_top_k_with_candis(self.sym_info['true_syms'], self.sym_info['false_syms'],candidate_diseases)

            t2_str =  self.get_candidate_dis(candi,is_en)


            cur_generate += t2_str
            generated_text = self.model_genrate(prompt+cur_generate)

            try:
                if is_en:
                    pattern = r"## Diagnostic confidence:\s*(.*)"
                else:
                    pattern = r"## 诊断置信度:\s*(.*)"
                match = re.search(pattern, generated_text, re.DOTALL)
                confidence_distribution = json.loads(match.group(1))
            except Exception as e:
                print('Error')
                print(e)
                print(confidence_distribution)
                return cur_generate + generated_text

            cur_generate += generated_text
            max_key = max(confidence_distribution, key=confidence_distribution.get)
            max_value = confidence_distribution[max_key]
            
            if max_value > self.confidence_threshold:
                # diagnosis
                if is_en:
                    t4_1 = '\n\n## Diagnosis:\n'
                else:
                    t4_1 = '\n\n## 做出诊断:\n'
                # print(t4_1,end='',flush = True)
                cur_generate += t4_1
                generated_text = self.model_genrate(prompt+cur_generate)
                cur_generate += generated_text

                self.history = self.history + [(query, cur_generate)]

            else:
                # Diagnosis
                if is_en:
                    t4_2 = '\n\n\nInadequate for diagnosis. Ask for symptoms:\n'
                else:
                    t4_2 = '\n\n\n为了更好的诊断疾病，我还需要了解您更多信息，请您回答我的问题:\n'
                # print(t4_2,end='',flush = True)
                cur_generate += t4_2
                generated_text = self.model_genrate(prompt+cur_generate)
                cur_generate += generated_text

                self.history = self.history + [(query, cur_generate)]

            return cur_generate, self.history, confidence_distribution
        # For other chat.
        else:
            self.history = self.history + [(query, cur_generate)]
            return cur_generate, self.history, {}
    

def main(args):
    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    
    chatbot = DiagnosisChatbot(args.model_dir,confidence_threshold = 0.5)

    pre_string = "DiagnosisGPT: Hello! I'm a large language model designed for disease diagnosis. Please tell me about your symptoms. Type 'clear' to reset the dialogue history, or 'stop' to end the session."
    print(pre_string)

    while True:
        query = input("\nUser: ")
        if query == "stop":
            break
        if query == "clear":
            chatbot.history = []
            os.system(clear_command)
            print(pre_string)
            continue
        
        print(f"DiagnosisGPT: ", end="", flush=True)
        for char in chatbot.chat(query):
            print(char,end='',flush = True)
        print('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/mntcephfs/data/med/zhanghongbo/yaojishi/cjy/ckpts/huatuo2_7B_re2_test/checkpoint-0-50/tfmr")
    args = parser.parse_args()
    main(args)