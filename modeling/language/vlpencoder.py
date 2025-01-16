# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from .build import register_model
from ..utils import configurable
from .LangEncoder import build_tokenizer, build_lang_encoder
from utils.prompt_engineering import prompt_engineering, get_prompt_templates
from ..utils import get_class_names

class LanguageEncoder(nn.Module):

    @configurable
    def __init__(
        self,
        tokenizer,
        tokenizer_type,
        lang_encoder,
        lang_projection,
        max_token_num,
        queue_operator,
        dataset_name
    ):
        super().__init__()
        # seg
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.lang_encoder = lang_encoder
        self.lang_proj = lang_projection
        self.max_token_num = max_token_num
        self.logit_scale = nn.Parameter(torch.ones([]))

        # add for prompt learner
        self.dataset_class_names = get_class_names(dataset_name)
        
        import clip
        device = "cpu" #if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=device)

        self.prompt_learner = PromptLearnerCoop(classnames=(self.dataset_class_names),clip_model=clip_model)
        self.PromptLearner_init = True #require initialize or not
        self.init_global = True
        self.init_local = True
        self.init_hard = True
        # captioning & retrieval
        for key, value in queue_operator.items():
            self.register_buffer(key, value)
            

    @classmethod
    def from_config(cls, cfg):
        # build up text encoder for seg
        tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        tokenizer_type = cfg['MODEL']['TEXT']['TOKENIZER']
        lang_encoder = build_lang_encoder(cfg['MODEL']['TEXT'], tokenizer, cfg['VERBOSE'])
        max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        
        dim_lang = cfg['MODEL']['TEXT']['WIDTH']
        dim_projection = cfg['MODEL']['DIM_PROJ']
        lang_projection = nn.Parameter(torch.empty(dim_lang, dim_projection))
        trunc_normal_(lang_projection, std=.02)

        dataset_name = cfg['DATASETS']['TRAIN'][0]

        # tested not working better      
        queue_operator = {}

        return {
            "tokenizer": tokenizer,
            "tokenizer_type": tokenizer_type,
            "lang_encoder": lang_encoder,
            "lang_projection": lang_projection,
            "max_token_num": max_token_num,
            "queue_operator": queue_operator,
            "dataset_name": dataset_name,
        }

    def get_text_embeddings(self, class_names, name='default', is_eval=False, add_bgd=False, prompt=True, norm=True, store_buffer=None):
        if not is_eval:
            if prompt:
                # randomly sample one template
                arbitary_concepts = [
                    prompt_engineering(class_names[label].replace('-other','').replace('-merged','').replace('-stuff',''), topk=10000, suffix='.') \
                    for label in range(len(class_names))
                ]
                if add_bgd:
                    arbitary_concepts.append("A background in coco.")
            else:
                arbitary_concepts = class_names
            
            input_ids = []
            attention_masks = []
            for txt in arbitary_concepts:
                tokens = self.tokenizer(
                    txt, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
                tokens['input_ids'].squeeze_()
                tokens['attention_mask'].squeeze_()

                input_ids.append(tokens['input_ids'])
                attention_masks.append(tokens['attention_mask'])

            arbitary_tokens = torch.stack(input_ids)
            arbitary_attention_masks = torch.stack(attention_masks)

            text_emb = self.forward_language((arbitary_tokens.cuda(), arbitary_attention_masks.cuda()), norm=norm)
            setattr(self, '{}_text_embeddings'.format(name), text_emb)
        else:
            with torch.no_grad():
                def extract_mean_emb(txts):
                    tokens = self.tokenizer(
                        txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                    )
                    clss_embedding = self.forward_language((tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()), norm=norm)
                    clss_embedding = clss_embedding.mean(dim=0)
                    clss_embedding /= clss_embedding.norm()
                    return clss_embedding

                templates = get_prompt_templates()
                clss_embeddings = []
                if prompt:
                    for clss in class_names:
                        txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in templates]
                        clss_embeddings.append(extract_mean_emb(txts))
                else:
                    for clss in class_names:
                        clss_embeddings.append(extract_mean_emb([clss]))

                if add_bgd:
                    txts = ["A background in coco."]
                    clss_embeddings.append(extract_mean_emb(txts))

                text_emb = torch.stack(clss_embeddings, dim=0)
                setattr(self, '{}_text_embeddings'.format(name), text_emb)

    def reset_text_embeddings(self, name='default'):
        pass

    def get_text_token_embeddings(self, txts, name='default', token=False, norm=False):
        if not token:
            tokens = self.tokenizer(
                txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
            )
            tokens = {key: value.cuda() for key, value in tokens.items()}
        else:
            tokens = txts
        token_emb, class_emb = self.forward_language_token((tokens['input_ids'], tokens['attention_mask']), norm=norm)
        ret = {"tokens": tokens,
                "token_emb": token_emb,
                "class_emb": class_emb,}
        setattr(self, '{}_token_embeddings'.format(name), ret)
        return ret

    def forward_language(self, texts, norm=True):
        x = self.lang_encoder(*texts)
        x = x['last_hidden_state']

        if self.tokenizer_type == 'clip':
            x = x[torch.arange(x.size(0)), texts[0].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.lang_proj
        if norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        return x
    
    def forward_language_token(self, texts, norm=False):
        x = self.lang_encoder(*texts)
        token_x = x['last_hidden_state']

        if self.tokenizer_type == 'clip':
            class_x = token_x[torch.arange(token_x.size(0)), texts[0].argmax(dim=-1)]
        else:
            class_x = token_x[:, 0]

        class_x = class_x @ self.lang_proj
        token_x = token_x @ self.lang_proj

        if norm:
            class_x = class_x / (class_x.norm(dim=-1, keepdim=True) + 1e-7)
            token_x = token_x / (token_x.norm(dim=-1, keepdim=True) + 1e-7)

        return token_x, class_x
    
    def compute_similarity(self, v_emb, name='default', fake=False, ema=False, st=False, hard=False):
        if fake:
            return None
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        if ema:
            t_emb = getattr(self, '{}_text_embeddings_ema'.format(name))
        elif st:
            t_emb = getattr(self, '{}_text_embeddings_st'.format(name))
        elif hard:
            t_emb = getattr(self, '{}_text_embeddings_hard'.format(name))
        else:
            t_emb = getattr(self, '{}_text_embeddings'.format(name))
        output = self.logit_scale.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        return output

    def forward_language_token_embedding(self, token_embeddings, tokenized_prompts, norm=True):
        x = self.lang_encoder.forward_token_embedding(token_embeddings)
        x = x['last_hidden_state']

        if self.tokenizer_type == 'clip':
            x = x[torch.arange(x.size(0)), tokenized_prompts.argmax(dim=-1)]
        else:
            x = x[:, 0]
        x = x @ self.lang_proj
        if norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        return x

    #add for coop prompt learner
    def get_text_embeddings_coop(self, class_names, name='default', is_eval=False, add_bgd=False, prompt=True, norm=True, ema=True):
        if self.PromptLearner_init:
            if is_eval:
                self.PromptLearner_init = False
            else:
                with torch.no_grad():
                    embedding = self.lang_encoder.token_embedding(self.prompt_learner.promptLearner.tokenized_prompts.cuda())
                    self.prompt_learner.promptLearner.token_prefix = embedding[:, :1, :]  # SOS
                    self.prompt_learner.promptLearner.token_suffix = embedding[:, 1 + self.prompt_learner.promptLearner.n_ctx :, :]  # CLS, EOS
                    self.prompt_learner.promptLearner.ctx.data = embedding[:, 1:1 + self.prompt_learner.promptLearner.n_ctx, :].detach().clone() # # #### initial all context as class names

                self.PromptLearner_init = False
        
        if not is_eval:
            text_emb = self.forward_language_token_embedding(self.prompt_learner(), self.prompt_learner.promptLearner.tokenized_prompts)
            setattr(self, '{}_text_embeddings'.format(name), text_emb)
        else:
            with torch.no_grad():
                text_emb = self.forward_language_token_embedding(self.prompt_learner(), self.prompt_learner.promptLearner.tokenized_prompts)
                setattr(self, '{}_text_embeddings'.format(name), text_emb)

    

    def update_ema_prompt(self, name='default'):

        if self.init_global: #first initialize
            emb = getattr(self, '{}_text_embeddings'.format(name))
            setattr(self, '{}_text_embeddings_ema'.format(name), emb)
            self.init_global = False
        else: #normal update
            old_global_emb = getattr(self, '{}_text_embeddings_ema'.format(name))
            new_global_emb = 1/2 * (self.memory_local[-1] + self.memory_hard[-1])
            global_emb = old_global_emb.clone() * 0.99 + new_global_emb.clone() * (1 - 0.99)
            setattr(self, '{}_text_embeddings_ema'.format(name), global_emb)

        # self.prompt_learner.promptLearner.ctx_ema.data = self.prompt_learner.promptLearner.ctx_ema.data.clone() * 0.9 + self.prompt_learner.promptLearner.ctx.data.clone() * (1. - 0.9)

    
    def update_st_memory(self, entropy, name='default'):

        if self.init_local: #first initialize
            emb = getattr(self, '{}_text_embeddings'.format(name))
            self.memory_local = emb.repeat(32,1,1)
            self.entropy = entropy.expand(32)
            self.init_local = False
        else:
            # Normal update
            new_local_emb = getattr(self, '{}_text_embeddings'.format(name))
            self.memory_local[1:] = self.memory_local[:-1].clone()
            self.memory_local[:1] = new_local_emb.clone()
            # Update entropy
            new_entropy = torch.cat([entropy.unsqueeze(0),self.entropy[:-1]], dim=0)
            self.entropy = new_entropy


    def update_st_prompt(self, name='default'):

        local_emb = self.memory_local.clone().mean(0)
        setattr(self, '{}_text_embeddings_st'.format(name), local_emb)

    
    def update_hard_memory(self, name='default'):

        if self.init_hard: #first initialize
            emb = getattr(self, '{}_text_embeddings'.format(name))
            self.memory_hard = emb.repeat(32,1,1)
            self.init_hard = False
        else:
            # Normal update
            _, indices = torch.topk(self.entropy, 16, dim=0)
            new_hard_emb = self.memory_local[indices].mean(0)
            self.memory_hard[1:] = self.memory_hard[:-1].clone()
            self.memory_hard[:1] = new_hard_emb.clone()


    def update_hard_prompt(self, name='default'):

        hard_emb = self.memory_hard.clone().mean(0)
        setattr(self, '{}_text_embeddings_hard'.format(name), hard_emb)





@register_model
def get_language_model(cfg, **kwargs):
    return LanguageEncoder(cfg)


#add Class for coop prompt learner
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = "a_photo_of_a"
        class_specific_context = True
        # class_specific_context = False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution


        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if class_specific_context: # cfg.TRAINER.COOP.CSC:  # class-specific context (False or True)
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number parameters: {ctx_vectors.shape}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"


    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts

class PromptLearnerCoop(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.promptLearner = PromptLearner(classnames, clip_model)

    def forward(self):
        prompts = self.promptLearner()
        return prompts