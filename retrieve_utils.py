import sys
sys.path += ['./']
import os
import faiss
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import torch
from transformers import AutoModel, AutoTokenizer
import json
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, encoder_model_path,cudaid = 0):
        super(encoder, self).__init__()
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_path, trust_remote_code=True)
        self.encoder_model = AutoModel.from_pretrained(encoder_model_path, trust_remote_code=True)
        self.output_embedding_size = self.encoder_model.config.hidden_size
        self.index = None
        self.disease = None
        self.diseaseid2id = None
        self.disease2id = None
        # if cudaid > -1:
        #     self.encoder_model = self.encoder_model.to(f'cuda:{cudaid}')
        self.norm = torch.nn.LayerNorm(self.output_embedding_size).to(self.encoder_model.device)
        self.max_seq_length = 128
        self.exclude_ids = []

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return self.norm(s / d)
    

    def get_query(self, symptoms):
        prompt = ', '.join(symptoms)
        return prompt
    
    def search_disease(self, symptoms):
        query_embeddings = self.encode(self.get_query(symptoms))
        values,ids = self.index.search(query_embeddings, 100)
        ids = ids.tolist()[0]
        values = values.tolist()[0]
        return ids, values

    def find_top_k(self, true_syms, false_syms=None, k=3):
        disease_rank, values = self.search_disease(true_syms)
        
        if false_syms is not None:
            f_disease_rank, f_values = self.search_disease(true_syms + false_syms)
            
            # Recalculate the disease score
            false_weight = 0.3
            for i in range(len(disease_rank)):
                if disease_rank[i] in f_disease_rank:
                    values[i] = values[i] - false_weight * f_values[f_disease_rank.index(disease_rank[i])]
                else:
                    values[i] = values[i] - false_weight * f_values[-1]

            # Rerank the scores of the disease
            values = np.array(values)
            disease_rank = np.array(disease_rank)
            # Inreverse order
            sort_index = np.argsort(-values)
            disease_rank = disease_rank[sort_index].tolist()
            values = values[sort_index].tolist()
        # k_disease_rank = disease_rank[:k]

        top_k = []
        for i in range(len(disease_rank)):
            if disease_rank[i] in self.exclude_ids:
                continue
            top_dis = self.disease[disease_rank[i]]
            top_dis["score"] = values[i]
            top_dis["rank"] = i + 1
            top_k.append(top_dis)
            if len(top_k) >= k:
                break
        return top_k
    

    def search_disease_with_candis(self,symptoms,candidate_id):
        query_embeddings = self.encode(self.get_query(symptoms))

        specific_vectors = np.vstack([self.index.reconstruct(int(id)) for id in candidate_id])
        
        similarities = np.dot(specific_vectors, query_embeddings.T).flatten()

        sorted_indices = np.argsort(-similarities)
        
        ids = [candidate_id[i] for i in sorted_indices]
        values = [similarities[i] for i in sorted_indices]

        return ids, values
    
    
    def find_top_k_with_candis(self, true_syms, false_syms=None, candidate_diseases = [], k=3):
        if not candidate_diseases or len(candidate_diseases) == 0:
            return self.find_top_k(true_syms, false_syms, k=3)

        candidate_id = [ self.disease2id[x] for x in candidate_diseases]

        disease_rank, values = self.search_disease_with_candis(true_syms,candidate_id)
        
        if false_syms is not None:
            f_disease_rank, f_values = self.search_disease_with_candis(true_syms + false_syms,candidate_id)

            # Recalculate the disease score
            false_weight = 0.3
            for i in range(len(disease_rank)):
                if disease_rank[i] in f_disease_rank:
                    values[i] = values[i] - false_weight * f_values[f_disease_rank.index(disease_rank[i])]
                else:
                    values[i] = values[i] - false_weight * f_values[-1]

            # Rerank the scores of the disease
            values = np.array(values)
            disease_rank = np.array(disease_rank)
            # Inreverse order
            sort_index = np.argsort(-values)
            disease_rank = disease_rank[sort_index].tolist()
            values = values[sort_index].tolist()

        # disease_rank = disease_rank[:k]
        top_k = []
        for i in range(len(disease_rank)):
            if disease_rank[i] in self.exclude_ids:
                continue
            top_dis = self.disease[disease_rank[i]]
            top_dis["score"] = values[i]
            top_dis["rank"] = i + 1
            top_k.append(top_dis)
            if len(top_k) >= k:
                break
        return top_k
        
    
    def encode(self, query):
        # input_tensors = self.encoder_tokenizer(query, return_tensors='pt')
        input_tensors = self.encoder_tokenizer(query, max_length=self.max_seq_length, truncation=True, return_tensors='pt', padding=True, add_special_tokens=True)

        input_ids = input_tensors['input_ids'].to(self.encoder_model.device)
        attention_mask = input_tensors['attention_mask'].to(self.encoder_model.device)
        with torch.no_grad():
            output = self.encoder_model(input_ids=input_ids.to(self.encoder_model.device),
                                        attention_mask=attention_mask.to(self.encoder_model.device))
            logits = self.masked_mean(output[0][:, :], attention_mask[:, :],
                                      ).detach().cpu().numpy()
        return logits
    
    def grad_encode(self,input_ids,attention_mask):
        input_ids = input_ids.to(self.encoder_model.device)
        attention_mask = attention_mask.to(self.encoder_model.device)
        output = self.encoder_model(input_ids=input_ids,
                                        attention_mask=attention_mask)
        logits = self.masked_mean(output[0][:, :], attention_mask[:, :])
        return logits
    
    def save_pretrained(self, save_directory):
        """Saves the model and tokenizer to the specified directory."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the model
        self.encoder_model.save_pretrained(save_directory)
        # Save the tokenizer
        self.encoder_tokenizer.save_pretrained(save_directory)

    def forward(self, 
            # input_query_ids, query_attention_mask,
            query_embs,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None):
        # query_embs = self.grad_encode(input_query_ids, query_attention_mask)
        doc_embs = self.grad_encode(input_doc_ids, doc_attention_mask)
        other_doc_embs = self.grad_encode(other_doc_ids, other_doc_attention_mask)
        
        with autocast(enabled=False):

            batch_scores = torch.matmul(query_embs, doc_embs.T)
            other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
            score = torch.cat([batch_scores, other_batch_scores], dim=-1)

            if rel_pair_mask is not None and hard_pair_mask is not None:
                rel_pair_mask = rel_pair_mask.to(self.encoder_model.device)
                hard_pair_mask = hard_pair_mask.to(self.encoder_model.device)
                rel_pair_mask = rel_pair_mask + torch.eye(score.size(0), dtype=score.dtype, device=score.device)
                mask = torch.cat([rel_pair_mask, hard_pair_mask], dim=-1)
                score = score.masked_fill(mask==0, -10000)
            
            labels = torch.arange(start=0, end=score.shape[0],
                                  dtype=torch.long, device=score.device)
            loss = F.cross_entropy(score, labels)
            acc = torch.sum(score.max(1).indices == labels) / score.size(0)

            return (loss, acc)

def load_dataset(preprocess_dir):
    with open(preprocess_dir) as f:
        datas = json.load(f)
    return datas


def index_retrieve(index, query_embeddings, topk, batch=None):
    print("Query Num", len(query_embeddings))
    start = timer()
    if batch is None:
        _, nearest_neighbors = index.search(query_embeddings, topk)
    else:
        query_offset_base = 0
        pbar = tqdm(total=len(query_embeddings))
        nearest_neighbors = []
        while query_offset_base < len(query_embeddings):
            batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base+ batch]
            batch_nn = index.search(batch_query_embeddings, topk)[1]
            nearest_neighbors.extend(batch_nn.tolist())
            query_offset_base += len(batch_query_embeddings)
            pbar.update(len(batch_query_embeddings))
        pbar.close()

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms")
    return nearest_neighbors


def construct_flatindex_from_embeddings(embeddings = None, ids=None):
    dim = embeddings.shape[1]
    print('embedding shape: ' + str(embeddings.shape))
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if embeddings is not None:
        if ids is not None:
            ids = ids.astype(np.int64)
            print(ids.shape, ids.dtype)
            index = faiss.IndexIDMap2(index)
            index.add_with_ids(embeddings, ids)
        else:
            index.add(embeddings)
        return index


gpu_resources = []

def convert_index_to_gpu(index, faiss_gpu_index, useFloat16=False):
    if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
        faiss_gpu_index = faiss_gpu_index[0]
    if isinstance(faiss_gpu_index, int):
        res = faiss.StandardGpuResources()
        res.setTempMemory(500*1024*1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = useFloat16
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
    else:
        global gpu_resources
        if len(gpu_resources) == 0:
            import torch
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                # res.setTempMemory(256*1024*1024)
                res.setTempMemory(4*1024*1024)
                gpu_resources.append(res)

        assert isinstance(faiss_gpu_index, list)
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = useFloat16
        for i in faiss_gpu_index:
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    return index