import logging
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


from transformers.models.bert.modeling_bert import (
    BertConfig, 
    BertPreTrainedModel, 
    BertEncoder, 
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertLMHeadModel,
    )

logger = logging.getLogger(__name__)


class TcplpConfig(BertConfig):

    def __init__(self,
                 max_token_num: int = 2048,
                 max_item_embeddings: int = 32,  # 1 for <s>, 50 for items
                 max_attr_length: int = 8,
                 pooler_type: str = 'cls',
                 temp: float = 0.05,
                 item_num: int = 0,
                 finetune_negative_sample_size: int = 0,
                 share_p: bool = True,
                 special_p: bool = True,
                 p_content: int = 4,
                 p_atten: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.max_token_num = max_token_num
        self.max_item_embeddings = max_item_embeddings
        self.max_attr_length = max_attr_length
        self.pooler_type = pooler_type
        self.temp = temp

        # finetune config
        self.item_num = item_num
        self.finetune_negative_sample_size = finetune_negative_sample_size

        self.share_p = share_p
        self.special_p = special_p
        self.p_content  = p_content
        self.p_atten = p_atten

@dataclass
class TcplpPretrainingOutput:
    cl_correct_num: float = 0.0
    cl_total_num: float = 1e-5
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx



class TcplpEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config: TcplpConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.item_position_embeddings = nn.Embedding(config.max_item_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        item_position_ids=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        item_position_embeddings = self.item_position_embeddings(item_position_ids)

        embeddings = inputs_embeds + item_position_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TcplpPooler(nn.Module):
    def __init__(self, config: TcplpConfig):
        super().__init__()
        self.pooler_type = config.pooler_type

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        output = None
        if self.pooler_type == 'cls':
            output = hidden_states[:, 0]
        else:
            raise NotImplementedError
        return output


class TcplpModel(BertPreTrainedModel):
    def __init__(self, config: TcplpConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = TcplpEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = TcplpPooler(config)

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        item_position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)


        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            item_position_ids = item_position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, config: TcplpConfig):
        super().__init__()
        self.temp = config.temp
    def forward(self, x, y):
        return torch.matmul(x, y.T) / self.temp


class TcplpForPretraining(BertPreTrainedModel):
    def __init__(self, config: TcplpConfig):
        super().__init__(config)
        self.config = config
        self.bert = TcplpModel(config)
      
        self.share_prompts = nn.Parameter(torch.randn(config.p_content, config.hidden_size))
        nn.init.xavier_uniform_(self.share_prompts)
        self.attn_layer = nn.MultiheadAttention(config.hidden_size, num_heads=4)

        self.special_prompts = nn.Parameter(torch.randn(config.p_content, config.hidden_size))
        nn.init.xavier_uniform_(self.special_prompts)
        self.attn_layer2 = nn.MultiheadAttention(config.hidden_size, num_heads=4)
        self.netx = netx()

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.sim = Similarity(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_prompt(self, seq_out, share_prompts, special_prompts):

        share_prompts, _ = self.attn_layer(seq_out, share_prompts, share_prompts)
        special_prompts, _ = self.attn_layer2(seq_out, special_prompts, special_prompts)

        return share_prompts, special_prompts

    def forward(
            self,
            input_ids_a: Optional[torch.Tensor] = None,
            attention_mask_a: Optional[torch.Tensor] = None,
            global_attention_mask_a: Optional[torch.Tensor] = None,
            item_position_ids_a: Optional[torch.Tensor] = None,
            input_ids_b: Optional[torch.Tensor] = None,
            attention_mask_b: Optional[torch.Tensor] = None,
            global_attention_mask_b: Optional[torch.Tensor] = None,
            item_position_ids_b: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids_a.size(0)

        outputs_a = self.bert(
            input_ids_a,
            attention_mask=attention_mask_a,
            head_mask=head_mask,
            position_ids=position_ids,
            item_position_ids=item_position_ids_a,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        outputs_b = self.bert(
            input_ids_b,
            attention_mask=attention_mask_b,
            head_mask=head_mask,
            position_ids=position_ids,
            item_position_ids=item_position_ids_b,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        z1 = outputs_a.pooler_output
        z2 = outputs_b.pooler_output

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]

            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2

            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
        

        share_prompts = self.share_prompts # (content, dim)
        special_prompts = self.special_prompts # (content, dim)
        share_prompt_z1, special_prompt_z1 = self.get_prompt(z1, share_prompts, special_prompts) # (batch, dim)
        output_z1 = torch.cat((z1, share_prompt_z1, special_prompt_z1), -1) # (batch, dim*3)
        share_prompt_z2, special_prompt_z2 = self.get_prompt(z2, share_prompts, special_prompts) # (batch, dim)
        output_z2 = torch.cat((z2, share_prompt_z2, special_prompt_z2), -1) # (batch, dim*3)

        output_z1 = self.netx(output_z1)
        output_z1 = self.LayerNorm(output_z1)
        output_z2 = self.netx(output_z2)
        output_z2 = self.LayerNorm(output_z2)         

        cos_sim = self.sim(output_z1, output_z2)
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)

        correct_num = (torch.argmax(cos_sim, 1) == labels).sum()

        return TcplpPretrainingOutput(
            loss=loss,
            logits=cos_sim,
            cl_correct_num=correct_num,
            cl_total_num=batch_size,
            hidden_states=outputs_a.hidden_states,
            attentions=outputs_a.attentions,
        )

class netx(nn.Module):
    def __init__(self):
        super(netx,self).__init__()
        self.layer1 = nn.Linear(3*768, 2*768)
        self.layer2 = nn.Linear(2*768, 1*768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class TcplpForFinetune(BertPreTrainedModel):
    def __init__(self, config: TcplpConfig):
        super().__init__(config)

        self.bert = TcplpModel(config)
        self.sim = Similarity(config)

        self.share_prompts = nn.Parameter(torch.randn(config.p_content, config.hidden_size))
        nn.init.xavier_uniform_(self.share_prompts)
        self.attn_layer = nn.MultiheadAttention(config.hidden_size, num_heads=4)

        self.special_prompts = nn.Parameter(torch.randn(config.p_content, config.hidden_size))
        nn.init.xavier_uniform_(self.special_prompts)
        self.attn_layer2 = nn.MultiheadAttention(config.hidden_size, num_heads=4)
                
        self.netx = netx()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def init_item_embedding(self, embeddings: Optional[torch.Tensor] = None):
        self.item_embedding = nn.Embedding(num_embeddings=self.config.item_num, embedding_dim=self.config.hidden_size)
        if embeddings is not None:
            self.item_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
            print('Initalize item embeddings from vectors.')

    def similarity_score(self, pooler_output, share_prompt=None, special_prompt=None, candidates=None):
        if candidates is None:
            candidate_embeddings = self.item_embedding.weight.unsqueeze(0)  # (1, num_items, hidden_size)
        else:
            candidate_embeddings = self.item_embedding(candidates)  # (batch_size, candidates, hidden_size)

        candidate_embeddings = candidate_embeddings.squeeze(0) #[1,candidates,hidden_size] -> [candidates,hidden_size]

        share_prompt_label, special_prompt_label = self.get_prompt(candidate_embeddings, share_prompt, special_prompt) #[candidates,hidden_size]
        output_label = torch.cat((candidate_embeddings, share_prompt_label, special_prompt_label), -1)
        output_label = self.netx(output_label)
        output_label = self.LayerNorm(output_label)
      
        # output_label = output_label.unsqueeze(0) # (1, candidates, hidden_size)
        # pooler_output = pooler_output.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        return self.sim(pooler_output, output_label)


    def get_prompt(self, seq_out, share_prompts, special_prompts):
        share_prompts, _ = self.attn_layer(seq_out, share_prompts, share_prompts)
        special_prompts, _ = self.attn_layer2(seq_out, special_prompts, special_prompts)

        return share_prompts, special_prompts


    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                global_attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                item_position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                candidates: Optional[torch.Tensor] = None,  # candidate item ids
                labels: Optional[torch.Tensor] = None,  # target item ids
                ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooler_output = outputs.pooler_output  # (bs, hidden_size)

        share_prompts = None
        special_prompts = None

        share_prompt, special_prompt = self.get_prompt(pooler_output, share_prompts, special_prompts) # (batch, dim)
        output = torch.cat((pooler_output, share_prompt, special_prompt), -1) # (batch, dim*3)
        output = self.netx(output)
        output = self.LayerNorm(output)   
  
        if labels is None:
            return self.similarity_score(pooler_output=output, share_prompt=share_prompts, special_prompt=special_prompts, candidates=None)

        loss_fct = CrossEntropyLoss()

        if self.config.finetune_negative_sample_size <= 0:  ## using full softmax
            logits = self.similarity_score(pooler_output=output, share_prompt=share_prompts, special_prompt=special_prompts,candidates=None)
            loss = loss_fct(logits, labels)

        else:
            candidates = torch.cat((labels.unsqueeze(-1), torch.randint(0, self.config.item_num, size=(
                batch_size, self.config.finetune_negative_sample_size)).to(labels.device)), dim=-1)
            logits = self.similarity_score(pooler_output, candidates)
            target = torch.zeros_like(labels, device=labels.device)
            loss = loss_fct(logits, target)

        return loss


