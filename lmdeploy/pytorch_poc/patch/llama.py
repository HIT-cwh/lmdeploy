# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DeviceMesh
from transformers.modeling_outputs import BaseModelOutputWithPast

from lmdeploy.pytorch_poc.dist_utils import (colwise_parallelize_linear_fn,
                                             rowwise_parallelize_linear_fn)
from lmdeploy.pytorch_poc.patch.functional import (
    apply_rotary_pos_emb, attention_forward_with_paged_attention)
from lmdeploy.pytorch_poc.patch.triton_kernels import rms_norm_dynamic_quant, per_token_quant_int8_tri, swiglu_int8_tri, linear_dynamic_quant_int8_tri, silu_elem_mul_tri, per_channel_quant, linear_dynamic_quant_triton_op_fast


class LlamaAttention(nn.Module):
    """Rewrite module of LlamaAttention."""

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['q_proj', 'k_proj', 'v_proj']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['o_proj']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs[0])
        return outputs

    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        world_size: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite implementation of LlamaAttention.forward.

        Add continuous batching support. Add paged attention support. TP
        support.
        """
        assert not output_attentions
        context = self.context.context
        history_lengths = context.history_lengths

        def _rotary_emb_fn(query_states, key_states, value_states):
            max_seq_len = position_ids.size(-1)
            kv_seq_len = max_seq_len + max(history_lengths)
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)
            return query_states, key_states, value_states

        attn_output = attention_forward_with_paged_attention(
            hidden_states,
            history_lengths=history_lengths,
            block_offsets=context.block_offsets,
            num_heads=self.num_heads // world_size,
            num_kv_heads=self.num_key_value_heads // world_size,
            head_dim=self.head_dim,
            position_ids=position_ids,
            past_key_value=past_key_value,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            o_proj=self.o_proj,
            rotary_emb_fn=_rotary_emb_fn,
        )

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Rewrite of LlamaAttention.forward."""
        if self.context.use_origin:
            return self.origin_mod(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            assert use_cache
            world_size = 1
            if dist.is_initialized():
                world_size = dist.get_world_size()
            return self._contiguous_batching_forward_impl(
                hidden_states,
                position_ids,
                past_key_value,
                output_attentions,
                world_size=world_size,
            )


class LlamaMLP(nn.Module):

    @classmethod
    def _distribute_partition_fn(cls, mod_name: str, mod: nn.Module,
                                 device_mesh: DeviceMesh):
        """Distribution partition callback."""
        if mod_name in ['gate_proj', 'up_proj']:
            colwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)
        elif mod_name in ['down_proj']:
            rowwise_parallelize_linear_fn(mod,
                                          device_mesh=device_mesh,
                                          to_local=True)

    @classmethod
    def _distribute_output_fn(cls, outputs, device_mesh: DeviceMesh):
        """Distribution output hook."""
        dist.all_reduce(outputs)
        return outputs


class LlamaDecoderLayer(nn.Module):
    # @classmethod
    def from_torch(self):
        # config = mod.self_attn.config
        # layer_triton = cls(config)

        # layer_triton.input_layernorm.load_state_dict(mod.input_layernorm.state_dict())
        # layer_triton.post_attention_layernorm.load_state_dict(mod.post_attention_layernorm.state_dict())
        # layer_triton.self_attn.rotary_emb.load_state_dict(mod.self_attn.rotary_emb.state_dict())

        ########### msa #########
        q_proj_weight = self.self_attn.q_proj.weight
        q_proj_weight_quant, q_proj_scale = per_channel_quant(q_proj_weight, 8, 1e-5, torch.int8)
        del self.self_attn.q_proj.weight
        self.register_buffer('q_proj_weight_quant', q_proj_weight_quant)
        self.register_buffer('q_proj_scale', q_proj_scale)

        k_proj_weight = self.self_attn.k_proj.weight
        k_proj_weight_quant, k_proj_scale = per_channel_quant(k_proj_weight, 8, 1e-5, torch.int8)
        del self.self_attn.k_proj.weight
        self.register_buffer('k_proj_weight_quant', k_proj_weight_quant)
        self.register_buffer('k_proj_scale', k_proj_scale)

        v_proj_weight = self.self_attn.v_proj.weight
        v_proj_weight_quant, v_proj_scale = per_channel_quant(v_proj_weight, 8, 1e-5, torch.int8)
        del self.self_attn.v_proj.weight
        self.register_buffer('v_proj_weight_quant', v_proj_weight_quant)
        self.register_buffer('v_proj_scale', v_proj_scale)

        o_proj_weight = self.self_attn.o_proj.weight
        o_proj_weight_quant, o_proj_scale = per_channel_quant(o_proj_weight, 8, 1e-5, torch.int8)
        del self.self_attn.o_proj.weight
        self.register_buffer('o_proj_weight_quant', o_proj_weight_quant)
        self.register_buffer('o_proj_scale', o_proj_scale)

        ############ mlp ###########

        gate_proj_weight = self.mlp.gate_proj.weight
        gate_proj_weight_quant, gate_proj_scale = per_channel_quant(gate_proj_weight, 8, 1e-5, torch.int8)
        del self.mlp.gate_proj.weight
        self.register_buffer('gate_proj_weight_quant', gate_proj_weight_quant)
        self.register_buffer('gate_proj_scale', gate_proj_scale)

        up_proj_weight = self.mlp.up_proj.weight
        up_proj_weight_quant, up_proj_scale = per_channel_quant(up_proj_weight, 8, 1e-5, torch.int8)
        del self.mlp.up_proj.weight
        self.register_buffer('up_proj_weight_quant', up_proj_weight_quant)
        self.register_buffer('up_proj_scale', up_proj_scale)

        down_proj_weight = self.mlp.down_proj.weight
        down_proj_weight_quant, down_proj_scale = per_channel_quant(down_proj_weight, 8, 1e-5, torch.int8)
        del self.mlp.down_proj.weight
        self.register_buffer('down_proj_weight_quant', down_proj_weight_quant)
        self.register_buffer('down_proj_scale', down_proj_scale)

        # return layer_triton
    
    def self_attn_forward(
        self,
        hidden_states, 
        attention_mask, 
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,):
        eps = self.input_layernorm.variance_epsilon
        residual = hidden_states

        hidden_states_quant, rms_scale = rms_norm_dynamic_quant(
            hidden_states, self.input_layernorm.weight, eps)
        
        assert not output_attentions
        context = self.context.context
        history_lengths = context.history_lengths

        def _rotary_emb_fn(query_states, key_states, value_states):
            max_seq_len = position_ids.size(-1)
            kv_seq_len = max_seq_len + max(history_lengths)
            cos, sin = self.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)
            return query_states, key_states, value_states
        
        attn_output = attention_forward_with_paged_attention(
            hidden_states_quant,
            rms_scale=rms_scale,
            history_lengths=history_lengths,
            block_offsets=context.block_offsets,
            num_heads=self.self_attn.num_heads,
            num_kv_heads=self.self_attn.num_key_value_heads,
            head_dim=self.self_attn.head_dim,
            position_ids=position_ids,
            past_key_value=past_key_value,
            q_proj_weight_quant=self.q_proj_weight_quant,
            q_proj_scale=self.q_proj_scale,
            k_proj_weight_quant=self.k_proj_weight_quant,
            k_proj_scale=self.k_proj_scale,
            v_proj_weight_quant=self.v_proj_weight_quant,
            v_proj_scale=self.v_proj_scale,
            o_proj_weight_quant=self.o_proj_weight_quant,
            o_proj_scale=self.o_proj_scale,
            rotary_emb_fn=_rotary_emb_fn,
            residual=residual,
        )

        return attn_output, past_key_value
    
    def mlp_forward(self, hidden_states):
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states_shape[-1])
        M, K = hidden_states.shape
        N = self.gate_proj_weight_quant.shape[0]
        eps = self.post_attention_layernorm.variance_epsilon

        hidden_states_quant, rms_scale = rms_norm_dynamic_quant(
            hidden_states, self.post_attention_layernorm.weight, eps)
        
        # if (M <= 32 and ((K <= 8192 and N <= 13824) or
        #                 (K <= 5120 and N <= 14336) or (K <= 4096 and N <= 28672))
        #         or M <= 64 and ((K <= 8192 and N <= 11008) or
        #                         (K <= 5120 and N <= 14336) or
        #                         (K <= 4096 and N <= 14336)) or M <= 128 and
        #     ((K <= 8192 and N <= 7168) or (K <= 5120 and N <= 13824) or
        #     (K <= 4096 and N <= 14336)) or M <= 256 and
        #     ((K <= 8192 and N <= 5504) or (K <= 5120 and N <= 7168) or
        #     (K <= 4096 and N <= 7168)) or M <= 512 and
        #     ((K <= 8192 and N <= 2752) or (K <= 5120 and N <= 3584) or
        #     (K <= 4096 and N <= 3584))):
        #     out_silu = swiglu_int8_tri(
        #         hidden_states_quant, self.gate_proj_weight_quant, 
        #         self.up_proj_weight_quant, rms_scale, 
        #         self.gate_proj_scale, self.up_proj_scale, eps, 
        #         output_dtype=hidden_states.dtype)
        # else:
        out_gate_proj = linear_dynamic_quant_triton_op_fast(
            hidden_states_quant, self.gate_proj_weight_quant, 
            rms_scale, self.gate_proj_scale, output_dtype=hidden_states.dtype)
        out_up_proj = linear_dynamic_quant_triton_op_fast(
            hidden_states_quant, self.up_proj_weight_quant, 
            rms_scale, self.up_proj_scale, output_dtype=hidden_states.dtype)
        out_silu = silu_elem_mul_tri(out_gate_proj, out_up_proj)
        
        out_silu_quant, out_silu_scale = per_token_quant_int8_tri(out_silu, eps)
        out = linear_dynamic_quant_triton_op_fast(
            out_silu_quant, self.down_proj_weight_quant, out_silu_scale, 
            self.down_proj_scale, hidden_states, output_dtype=hidden_states.dtype)
        out = out.view(hidden_states_shape)
        return out
    
    def _contiguous_batching_forward_impl(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        world_size: int = 1,
    ):
        hidden_states, present_key_value = self.self_attn_forward(
            hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)

        hidden_states = self.mlp_forward(hidden_states)

        return (hidden_states, present_key_value)

    # def _contiguous_batching_forward_impl(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #     output_attentions: Optional[bool] = False,
    #     use_cache: Optional[bool] = False,
    #     world_size: int = 1,
    # ):
    #     residual = hidden_states

    #     hidden_states = self.input_layernorm(hidden_states)

    #     assert not output_attentions
    #     context = self.context.context
    #     history_lengths = context.history_lengths

    #     def _rotary_emb_fn(query_states, key_states, value_states):
    #         max_seq_len = position_ids.size(-1)
    #         kv_seq_len = max_seq_len + max(history_lengths)
    #         cos, sin = self.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
    #         query_states, key_states = apply_rotary_pos_emb(
    #             query_states, key_states, cos, sin, position_ids)
    #         return query_states, key_states, value_states
        
    #     attn_output = attention_forward_with_paged_attention(
    #         hidden_states,
    #         history_lengths=history_lengths,
    #         block_offsets=context.block_offsets,
    #         num_heads=self.self_attn.num_heads // world_size,
    #         num_kv_heads=self.self_attn.num_key_value_heads // world_size,
    #         head_dim=self.self_attn.head_dim,
    #         position_ids=position_ids,
    #         past_key_value=past_key_value,
    #         q_proj=self.self_attn.q_proj,
    #         k_proj=self.self_attn.k_proj,
    #         v_proj=self.self_attn.v_proj,
    #         o_proj=self.self_attn.o_proj,
    #         rotary_emb_fn=_rotary_emb_fn,
    #     )

    #     hidden_states = residual + attn_output

    #     # Fully Connected
    #     residual = hidden_states
    #     hidden_states = self.post_attention_layernorm(hidden_states)
    #     hidden_states = self.mlp(hidden_states)
    #     hidden_states = residual + hidden_states

    #     return (hidden_states, past_key_value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.context.use_origin:
            return self.origin_mod(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
        else:
            assert use_cache
            world_size = 1
            return self._contiguous_batching_forward_impl(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )


class LlamaModel(nn.Module):

    def _continuous_batching_forward(
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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite implementation of LlamaModel.forward."""
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)

        if use_cache is None:
            use_cache = self.config.use_cache

        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        assert (
            position_ids is not None
        ), 'position_ids can not be none when using continuous batching mode.'
        assert position_ids.dim() == 2

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Attention mask is not necessary in continuous batching
        attention_mask = None

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            past_key_value = (past_key_values[idx]
                              if past_key_values is not None else None)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Rewrite of LlamaModel.forward."""
        use_origin = self.context.use_origin
        if use_origin:
            # use origin model
            return self.origin_mod(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
        else:
            return self._continuous_batching_forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
