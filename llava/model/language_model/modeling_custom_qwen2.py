from transformers.modeling_outputs import BaseModelOutputWithPast

from typing import List, Optional, Tuple, Union
import torch
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache, logging
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.model.data_store import data
logger = logging.get_logger(__name__)


class PDropQwen2Model(Qwen2Model):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        ### change position_embeddings into a list for future pruning
        position_embeddings = list(position_embeddings) 

        PDrop_retention_ratio=0.15
        device = self.device
        system_token_length=data.system_token_length
        user_instruction_length=data.user_instruction_length


        p_artio=0.25
        if p_artio==0.25:
            
            # Pdrop_layers = [6 ,12, 18] 
            Pdrop_layers = 6
            PDrop_retention_ratio=0.15
        elif p_artio==0.15:
            Pdrop_layers = [4,8, 12]
            PDrop_retention_ratio=0.05

        ###########################
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

##########################################################################################
            else:
                if use_cache:

                    if hidden_states.shape[1] != 1:
                        # if decoder_layer.self_attn.layer_idx in Pdrop_layers:
                        if decoder_layer.self_attn.layer_idx == Pdrop_layers:

                            before_instruction_lenth=hidden_states.shape[1]-user_instruction_length-1  # 减去换行符
                            pruned_attention_mask = causal_mask
                            last_layer_attention = layer_outputs[1]
                            last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                            last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                            last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[system_token_length:before_instruction_lenth]
                            k = round(last_layer_attention_avg_last_tok_image.shape[0] * PDrop_retention_ratio)
                            _, top_indices = last_layer_attention_avg_last_tok_image.topk(k)
                            sorted_indices = top_indices.sort().values
                            top_attention_rank_index = sorted_indices + system_token_length
                            keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), top_attention_rank_index, torch.arange(before_instruction_lenth,hidden_states.shape[1],device=device)))
                            keep_indexs = keep_indexs.sort().values
                            new_seq_length = keep_indexs.shape[0]
                            hidden_states = hidden_states[:,keep_indexs,:] 
                            position_ids = keep_indexs.unsqueeze(0)
                            position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
                            position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]
                            cache_position = cache_position[:new_seq_length]

                        else:
                            pruned_attention_mask = causal_mask
                    else:
                        pruned_attention_mask = causal_mask
                    
                else:
                    raise NotImplementedError("fastv only support use_cache=True")
                # out_attn_layer_indices = [x-1 for x in Pdrop_layers]
                # if decoder_layer.self_attn.layer_idx in out_attn_layer_indices:
                
                #     output_attentions = True
                # else:
                #     output_attentions = False

                if decoder_layer.self_attn.layer_idx == Pdrop_layers - 1:
                    output_attentions = True
                else:
                    output_attentions = False

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=pruned_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            ### force output_attentions to be False(we store attn_weights by ourselves)
            output_attentions = False
            ### end

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    



class FastvQwen2Model(Qwen2Model):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        ### change position_embeddings into a list for future pruning
        position_embeddings = list(position_embeddings) 
        FASTV_k=3
        Fastv_retention_ratio=0.15
        device = self.device
        system_token_length=data.system_token_length
        user_instruction_length=data.user_instruction_length
        ###########################
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

##########################################################################################
            else:
                if use_cache:

                    if hidden_states.shape[1] != 1:

                        if decoder_layer.self_attn.layer_idx==FASTV_k:
                        
                            before_instruction_lenth=hidden_states.shape[1]-user_instruction_length-1  # 减去换行符
                            pruned_attention_mask = causal_mask
                            last_layer_attention = layer_outputs[1]
                            last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                            last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                            last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[system_token_length:before_instruction_lenth]
                            k = round(last_layer_attention_avg_last_tok_image.shape[0] * Fastv_retention_ratio)
                            _, top_indices = last_layer_attention_avg_last_tok_image.topk(k)
                            sorted_indices = top_indices.sort().values
                            top_attention_rank_index = sorted_indices + system_token_length
                            keep_indexs = torch.cat( (torch.arange(system_token_length,device=device), top_attention_rank_index, torch.arange(before_instruction_lenth,hidden_states.shape[1],device=device)))
                            keep_indexs = keep_indexs.sort().values
                            new_seq_length = keep_indexs.shape[0]
                            hidden_states = hidden_states[:,keep_indexs,:] 
                            position_ids = keep_indexs.unsqueeze(0)
                            position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
                            position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]
                            cache_position = cache_position[:new_seq_length]

                        else:
                            pruned_attention_mask = causal_mask

                    else:
                        pruned_attention_mask = causal_mask
                else:
                    raise NotImplementedError("fastv only support use_cache=True")
                if decoder_layer.self_attn.layer_idx == FASTV_k - 1:
                    output_attentions = True
                else:
                    output_attentions = False

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=pruned_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
###########################################################################################
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            ### force output_attentions to be False(we store attn_weights by ourselves)
            output_attentions = False
            ### end

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )