from typing import List
import random
import string
import json

import tqdm

import numpy as np

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import load_dataset

import stanza
import nltk


from src.logging_utils import info
from src.data_schemas import ClassificationItem
from src.data_utils import read_corpus
from src.stop_words import ukr_stop_words
from src.synonym_utils import get_all_synonyms, prepare_synonym_dict

from attack.discriminator import Discriminator


class AttackClassification:
    def __init__(self,
                 dataset_path,
                 num_labels,
                 target_model_path,
                 counter_fitting_embeddings_path,
                 counter_fitting_cos_sim_path,
                 output_dir,
                 output_json,
                 sim_score_threshold,
                 synonym_num,
                 perturb_ratio,
                 discriminator_checkpoint,
                 mode,
                 path_to_synonym_dict
                 ) -> None:
        self.dataset_path = dataset_path
        self.plm_id = target_model_path
        self.num_labels = num_labels

        self.plm = AutoModelForSequenceClassification.from_pretrained(self.plm_id, num_labels=self.num_labels).to(
            'cuda')
        # self.tokenizer = AutoTokenizer.from_pretrained(self.plm_id)
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        info("Model built!")

        # prepare synonym extractor
        # build dictionary via the embedding file
        self.idx2word = {}
        self.word2idx = {}
        self.counter_fitting_embeddings_path = counter_fitting_embeddings_path
        self.counter_fitting_cos_sim_path = counter_fitting_cos_sim_path
        self.cos_sim = None
        self.build_vocab()

        self.stop_words = None
        self.get_stopwords()

        self.discriminator = Discriminator(info)

        # 其他参数
        self.output_file = f"{output_dir}/{output_json}"
        self.sim_score_threshold = sim_score_threshold
        self.synonym_num = synonym_num
        self.perturb_ratio = perturb_ratio
        self.checkpoint = discriminator_checkpoint
        self.mode = mode

        self.pos_tagger = stanza.Pipeline(lang='uk', processors='tokenize,mwt,pos', verbose=False)

        self.random = False
        self.synonym_dict = prepare_synonym_dict(path_to_synonym_dict)

    def build_vocab(self):
        info('Building vocab...')
        with open(self.counter_fitting_embeddings_path, 'r') as f:
            next(f)  # TODO regenerate embeding file to remove this
            for line in f:
                word = line.split()[0]
                if word not in self.idx2word:
                    self.idx2word[len(self.idx2word)] = word
                    self.word2idx[word] = len(self.idx2word) - 1

        info('Building cos sim matrix...')
        self.cos_sim = np.load(self.counter_fitting_cos_sim_path)
        info("Cos sim import finished!")

    def read_data(self, length=368, target_col='label') -> List[ClassificationItem]:
        if self.mode == 'eval':
            # if True:
            texts, labels = read_corpus(self.dataset_path)
            data = list(zip(texts, labels))
            data = [ClassificationItem(words=x[0][:length], label=int(x[1])) for x in data]
        elif self.mode == 'train':
            # dataset = load_dataset(self.dataset_path)
            dataset = load_dataset('csv', data_files={'train': self.dataset_path})
            train = list(dataset['train'])[:5000]
            data = [ClassificationItem(words=x['text'].lower().split()[:length], label=int(x[target_col])) for x in
                    train]
        info("Data import finished!")
        return data

    def get_stopwords(self):
        self.stop_words = set(ukr_stop_words)

    def predictor(self, texts: List[str]):
        self.plm.eval()
        encode_data = self.tokenizer(
            text=texts,
            return_tensors='pt', max_length=512, truncation=True, padding=True
        ).to('cuda')
        logits = self.plm(**encode_data).logits
        probability = torch.nn.functional.softmax(logits, dim=-1)
        label = torch.argmax(logits, dim=-1)
        return probability, label

    def eval(self):
        """原本acc
        """
        data = self.read_data()
        acc = 0
        bar = tqdm.tqdm(data, leave=True, position=0, desc='Eval')
        for i, item in enumerate(bar):
            _, label = self.predictor([item])
            if label[0] == item.label:
                acc += 1
            bar.set_postfix({'acc': acc / (i + 1)})
        bar.close()

    def find_synonyms(self, words: list, k: int = 50, threshold=0.7):
        """ 从 cos sim 文件中获取同义词
        words: list [w1, w2, w3...]
        k: int  candidate num
        threshold: float  similarity
        return:  {word1: [(syn1, score1), (syn2, score2)]}  score: similarity

            > a = AttackClassifier()
            > print(a.find_synonyms(['love', 'good']))
            > {'love': [('adores', 0.9859314), ('adore', 0.97984385), ('loves', 0.96597964), ('loved', 0.915682), ('amour', 0.8290837), ('adored', 0.8212078), ('iike', 0.8113105), ('iove', 0.80925167), ('amore', 0.8067776), ('likes', 0.80531627)], 'good': [('alright', 0.8443118), ('buena', 0.84190375), ('well', 0.8350292), ('nice', 0.78931385), ('decent', 0.7794969), ('bueno', 0.7718264), ('best', 0.7566938), ('guten', 0.7464952), ('bonne', 0.74170655), ('opportune', 0.738624)]}

        """
        ids = [self.word2idx[word] for word in words if word in self.word2idx]
        words = [word for word in words if word in self.word2idx]
        sim_order = np.argsort(-self.cos_sim[ids, :])[:, 1:1 + k]
        sim_words = {}  # word: [(syn1, score1, syn2, score2....)]

        for idx, word in enumerate(words):
            sim_value = self.cos_sim[ids[idx]][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [self.idx2word[idx] for idx in sim_word]
            if len(sim_word):
                sim_words[word] = [(sim_word[i], sim_value[i]) for i in range(len(sim_word))]
        return sim_words

    def find_synonyms_new(self, words: list):
        sim_words = {}

        for word in words:
            word = word.lower()

            if word in ukr_stop_words:
                continue

            synonyms = get_all_synonyms(word, self.synonym_dict)

            synonyms = [synonym for synonym in synonyms if len(synonym.split(' ')) == 1]

            if len(synonyms) == 0:
                continue

            sim_words[word] = [(s, 1) for s in synonyms]

        return sim_words

    def encode_state(self, words, pool):
        input_ids = []
        state = []
        word2token = {}  # index --> index
        for i, w in enumerate(words):
            ids = self.discriminator.tokenizer(w, add_special_tokens=False)['input_ids']
            state += [0] * len(ids) if i in pool else [1] * len(ids)
            word2token[i] = (len(input_ids), len(input_ids) + len(ids))  # 左闭右开
            input_ids += ids
        encode_data = {'input_ids': torch.tensor([input_ids]).to('cuda')}
        # state_infos = torch.tensor(state).unsqueeze(dim=0).unsqueeze(dim=-1)
        # last_hidden_state = self.plm(**encode_data, output_hidden_states=True).hidden_states[-1]
        # state_infos = state_infos.expand(state_infos.shape[0], state_infos.shape[1], 768)
        # output = torch.cat((last_hidden_state.to('cpu'), state_infos), dim=-1)  # 拼接
        state_infos = torch.tensor(state).unsqueeze(dim=0).unsqueeze(dim=-1).to('cuda')
        # print(len(input_ids))
        logits = self.discriminator(state_infos, **encode_data)
        return state, word2token, logits

    def replace(self, origin_label, words: list, index, synonyms: dict, org_pos):
        word = words[index]
        syn = synonyms.get(word, [])
        texts = []
        syn_copy = []

        for s in syn:

            item = words[max(1, index - 50): index] + [s[0]] + words[index + 1: min(len(words) - 1, index + 50)]
            # item = words[1: index] + [s[0]] + words[index+1: -1]
            # adv_pos = self.check_pos(item)
            adv_pos = org_pos
            if adv_pos[index] == org_pos[index] or ({adv_pos[index], org_pos[index]} <= {'NOUN', 'VERB'}):
                text = " ".join(item)
                texts.append(text)
                syn_copy.append(s)
        if len(texts) == 0:
            return None
        probability = None
        # batch: 8

        for i in range(0, len(texts), 16):
            # print(texts[i*8: i*8 + 8])

            p, _ = self.predictor(texts[i: i + 16])
            if i == 0:
                probability = p
            else:
                probability = torch.cat((probability, p), 0)

        # probability, label = self.predictor_by_words(texts)

        scores = probability[:, origin_label]
        _, slices = scores.sort()
        return syn_copy[slices[0]]  # 在对应标签上最低的

    def do_discrimination(self, state, word2token, logits):
        token_index_list = []
        token_index_p = []
        pro = torch.nn.functional.softmax(logits, dim=-1)
        for t, v in enumerate(pro[0]):
            if state[t] == 0:
                continue
            else:
                token_index_list.append(t)
                token_index_p.append(v.item())

        if self.random:
            token_index = random.choice(token_index_list)
        else:
            _, slices = torch.tensor(token_index_p).sort(descending=True)  # 降序取最大
            token_index = token_index_list[slices[0]]

        word_index = 0
        for i in word2token:
            if word2token[i][0] <= token_index < word2token[i][1]:
                word_index = i
        return word_index, word2token[word_index][0], logits

    def attack(self, item: ClassificationItem, attempts: int = 50):
        org_text = " ".join(item.words)
        orig_probs, orig_labels = self.predictor([org_text])
        orig_label = orig_labels[0]
        res = {'success': 0, 'org_label': item.label, 'org_seq': org_text, 'adv_seq': org_text, 'change': []}

        # if item.label != orig_label:  # 本身不正确，不进行attack
        #     return res

        if np.abs(item.label - orig_label.cpu().numpy()) > 1:
            return res

        # 初始化环境
        pool = set()
        words = ['[CLS]'] + item.words + ['[SEP]']
        org_pos = self.check_pos(words)
        # synonyms = self.find_synonyms(words, k=self.synonym_num, threshold=self.sim_score_threshold)
        synonyms = self.find_synonyms_new(words)

        for key, w in enumerate(words):
            if w in self.stop_words or w in ['[CLS]', '[SEP]'] or w in string.punctuation or w not in synonyms:
                pool.add(key)

        origin_pool = pool.copy()
        origin_words = words.copy()

        with torch.enable_grad():
            self.discriminator.train()

            for parameter in self.plm.parameters():
                parameter.require_grad = False  # 冻结PLm

            optimizer = AdamW(self.discriminator.parameters(), lr=3e-6)
            bar = tqdm.trange(attempts, leave=True, position=0, desc='Attack')
            min_step = 100
            constant = 0

            for _ in bar:
                optimizer.zero_grad()
                step = 0
                words = origin_words.copy()

                pool = origin_pool.copy()
                change = []

                if constant > 25:
                    break

                while len(pool) < len(words):

                    orig_probs, orig_labels = self.predictor([" ".join(words[1:-1])])
                    state, word2token, logits = self.encode_state(words, pool)

                    word_index, token_index, logits = self.do_discrimination(state, word2token, logits)

                    # 修改状态池
                    pool.add(word_index)
                    syn = self.replace(orig_label, words, word_index, synonyms, org_pos)  # (word, sim_score)
                    # 进入victim model
                    words[word_index] = syn[0]
                    new_text = " ".join(words[1:-1])
                    attack_probs, attack_labels = self.predictor([new_text])
                    attack_label = attack_labels[0]
                    change.append([word_index, origin_words[word_index], words[word_index]])
                    sub = (orig_probs[0][orig_label] - attack_probs[0][orig_label]).item()

                    reward = sub  # 增加下降值

                    # 计算期望
                    pro = torch.nn.functional.softmax(logits, dim=-1)[0]
                    h = -torch.log(pro[token_index])

                    step += 1
                    perturb = len(change) / len(item.words)

                    if perturb > self.perturb_ratio:
                        constant += 1
                        self.random = True
                        loss = -torch.abs(h * reward)
                        loss.backward()
                        break

                    loss = h * reward
                    loss.backward()

                    if np.abs(attack_label.cpu().numpy() - orig_label.cpu().numpy()) > 1:
                        # if attack_label != orig_label:
                        self.random = False
                        constant = 0
                        if step <= min_step:
                            min_step = step

                            res['success'] = 2
                            res['adv_label'] = attack_label.item()
                            res['adv_seq'] = new_text
                            res['change'] = change
                            res['perturb'] = perturb
                        break
                optimizer.step()

                perturb = len(change) / len(item.words)

                bar.set_postfix({'step': step, "perturb": perturb})

        return res

    def check_pos_old(self, org_text):
        word_n_pos_list = nltk.pos_tag(org_text, tagset="universal")
        _, pos_list = zip(*word_n_pos_list)
        return pos_list

    def check_pos(self, org_text):
        text = " ".join(org_text)
        # Replace [CLS] and [SEP] with placeholders
        text = text.replace("[CLS]", "CLS_TOKEN").replace("[SEP]", "SEP_TOKEN")
        doc = self.pos_tagger(text)
        # Process POS tagging
        results = [i.upos for j in range(len(doc.sentences)) for i in doc.sentences[j].words]
        return results

    def attack_eval(self, item: ClassificationItem):
        org_text = " ".join(item.words)
        _, orig_labels = self.predictor([org_text])
        orig_label = orig_labels[0]
        res = {'success': 0, 'org_label': item.label, 'org_seq': org_text, 'adv_seq': org_text, 'change': []}
        if item.label != orig_label:  # 本身不正确，不进行attack
            return res

        res['success'] = 1
        # 初始化环境
        pool = set()
        words = ['[CLS]'] + item.words + ['[SEP]']
        org_pos = self.check_pos(words)
        synonyms = self.find_synonyms(words, k=self.synonym_num, threshold=self.sim_score_threshold)
        for key, w in enumerate(words):
            if w in self.stop_words or w in ['[CLS]', '[SEP]'] or w in string.punctuation or w not in synonyms:
                pool.add(key)

        change = []
        origin_words = words.copy()
        step = 0
        while len(pool) < len(words):
            encode_state, state, word2token = self.encode_state(words, pool)
            word_index, _, _ = self.do_discrimination(encode_state, state, word2token)
            pool.add(word_index)
            syn = self.replace(orig_label, words, word_index, synonyms, org_pos)  # (word, sim_score)
            # 进入victim model
            words[word_index] = syn[0]
            new_text = " ".join(words[1:-1])
            _, attack_labels = self.predictor([new_text])
            attack_label = attack_labels[0]
            change.append([word_index, origin_words[word_index], words[word_index]])
            step += 1
            perturb = len(change) / len(item.words)
            # memory.append([word_index])
            if perturb > self.perturb_ratio:
                break
            if (attack_label != orig_label) and (perturb < self.perturb_ratio):
                res['success'] = 2
                res['adv_label'] = attack_label.item()
                res['adv_seq'] = new_text
                res['change'] = change
                res['perturb'] = perturb
                break
        return res

    def run(self):
        data = self.read_data(target_col='label')
        bar = tqdm.tqdm(data, position=1, leave=True, desc='Run attack')
        acc = 0
        attack_total = 0
        perturb_total = 0
        perturb = 0.0
        ans = []
        attack_func = self.attack
        if self.mode == 'eval':
            self.discriminator = Discriminator(self.checkpoint)
            self.discriminator.eval()
            attack_func = self.attack_eval
        for i, item in enumerate(bar):
            try:
                res = attack_func(item, 50)
            except Exception as e:
                print(f"{e}")
                res = {'success': 1, 'org_label': item.label, 'org_seq': " ".join(item.words),
                       'adv_seq': " ".join(item.words), 'change': []}
            ans.append(res)
            if res['success'] == 1:
                acc += 1
                attack_total += 1
            elif res['success'] == 2:
                perturb_total += 1
                perturb += res['perturb']
                attack_total += 1
            if perturb_total and attack_total:
                bar.set_postfix({'acc': acc / (i + 1), 'perturb': perturb / perturb_total,
                                 'attack_rate': perturb_total / attack_total})
            if i % 100 == 0 and self.mode == 'train':
                self.discriminator.saveModel(self.checkpoint)
            if i % 500 == 0 and self.mode == 'train':
                json.dump(ans, open(self.output_file, 'w'))

        json.dump(ans, open(self.output_file, 'w'))
        print(
            f"acc: {acc / len(data)}\t perturb: {perturb / perturb_total}\t attack_rate: {perturb_total / attack_total}")
        if self.mode == 'train':
            self.discriminator.saveModel(self.checkpoint)
