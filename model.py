import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Sequence, List, Tuple, Union, Any
from torch.nn import CrossEntropyLoss
from transformers import CLIPVisionModel, Blip2QFormerConfig, Blip2Config, Blip2VisionModel, Blip2QFormerModel, LlamaForCausalLM
from transformers.modeling_outputs import ModelOutput
from transformers.models.llama.modeling_llama import _make_causal_mask
from stable_diffusion import decode_with_sdxl
from einops import rearrange

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, tgt_seq_len, src_seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, tgt_len, src_len = mask.size()

    expanded_mask = mask[:, None, :, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

class MIMOutputWithPast(ModelOutput):
    
    logits: torch.FloatTensor = None
    past_key_values: Tuple[Tuple[torch.FloatTensor]] = None

class MIMModel(nn.Module):
    
    def __init__(self, args):
        super(MIMModel, self).__init__()

        self.args = args
        
        self.language_model = LlamaForCausalLM.from_pretrained(args.language_model, low_cpu_mem_usage=True)
        vision_model_class = Blip2VisionModel if "blip2" in args.vision_model else CLIPVisionModel
        self.vision_model = vision_model_class.from_pretrained(args.vision_model, low_cpu_mem_usage=True)
        
        # vision --> text qformer
        qformer_config = Blip2QFormerConfig(
            hidden_size=args.qformer_hidden_size, 
            intermediate_size=args.qformer_intermediate_size,
            num_hidden_layers=args.num_qformer_hidden_layers,
            num_attention_heads=args.num_qformer_attention_heads,
            encoder_hidden_size=self.vision_model.config.hidden_size,
            )
        self.query_tokens = nn.Parameter(torch.zeros(1, args.num_query_tokens, args.qformer_hidden_size))
        self.qformer = Blip2QFormerModel(qformer_config)
        self.qformer_projection = nn.Linear(qformer_config.hidden_size, self.language_model.config.hidden_size)
         
        # support multiple caption generation
        if args.train or args.compute_loss:
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask

    def get_token_embeds(
        self,
        input_ids,
    ):
        token_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        return token_embeds

    def get_image_embeds(
        self,
        input_images,
        dtype=None,
    ):
        # image_embeds: num_input_images x 3 x 224 x 224
        num_input_images = input_images.shape[0]
        image_embeds = self.vision_model(
            pixel_values=input_images if dtype is None else input_images.to(dtype),
        ).last_hidden_state

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        query_tokens = self.query_tokens.expand(num_input_images, -1, -1)
        image_embeds = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
        ).last_hidden_state
        # [num_input_images, num_query_tokens, language_model_hidden_size]
        image_embeds = self.qformer_projection(image_embeds)

        return image_embeds

    def get_inputs_embeds(
        self,
        input_ids,
        input_images=None,
        input_image_index=None,
    ):
        bsz = input_ids.shape[0]
        
        text_embeds = self.get_token_embeds(input_ids)
        
        if input_images is None:
            inputs_embeds = text_embeds
        
        else:     
            # step 1: forward the images through the vision encoder
            num_input_image = input_images.shape[1]
            input_images = rearrange(input_images, "bsz n a b c -> (bsz n) a b c")
            
            image_embeds = self.get_image_embeds(input_images, text_embeds.dtype)
            
            # [bsz*num_input_image, ...] --> [bsz, num_input_image, ...]
            image_embeds = rearrange(image_embeds, "(bsz n) a b -> bsz n a b", n=num_input_image)
            
            # update image embeds
            inputs_embeds = torch.scatter(
                input=text_embeds, dim=1, 
                index=input_image_index.unsqueeze(-1).expand(-1, -1, text_embeds.shape[-1]), 
                src=image_embeds.view(bsz, -1, image_embeds.shape[-1]))
                                       
        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        input_images=None,
        input_image_index=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
    ):

        if inputs_embeds is None:
            inputs_embeds = self.get_inputs_embeds(
                input_ids=input_ids,
                input_images=input_images,
                input_image_index=input_image_index,
            )
        
        # step 4: forward the input through the language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=use_cache,
        )        

        if not self.args.train and not self.args.compute_loss:
            return MIMOutputWithPast(
                    logits=outputs.logits,
                    past_key_values=outputs.past_key_values,
            )
        
        return outputs.loss

        
    def prepare_inputs_for_generation(
        self, 
        input_ids=None, 
        input_images=None, 
        input_image_index=None, 
        past_key_values=None, 
    ):
        if past_key_values:
            if input_images is not None:
                inputs_embeds = self.get_image_embeds(input_images=input_images)
            else:
                inputs_embeds = self.get_token_embeds(input_ids=input_ids[:, -1:])   
        else:
            inputs_embeds = self.get_inputs_embeds(
                input_ids=input_ids,
                input_images=input_images,
                input_image_index=input_image_index,
            )

        return {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values
        return model_kwargs
    
    @torch.no_grad()
    def cache_generation(
        self, 
        input_ids, 
        tokenizer, 
        image_processor,
        input_images=None,
        input_image_index=None, 
        caption_start_id=None,
        caption_end_id=None,
        sd_base=None, 
        sd_refiner=None,
        generator=None, 
        generate_image=False,
        max_output_length=256,
        top_p=None,
        temperature=1.0,
        **model_kwargs,
        ):
                
        image_list = []   
        caption_list = []
        for _ in range(max_output_length):
            
            model_inputs = self.prepare_inputs_for_generation(input_ids, input_images, input_image_index, **model_kwargs)
            outputs = self(**model_inputs, use_cache=True)
            next_token_scores = outputs.logits[:, -1, :]
            
            if top_p is not None:
                # top-p sampling
                next_token_scores = next_token_scores / temperature
                next_token_scores = topp_logits_filter(next_token_scores, top_p)
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = next_token_scores.argmax(-1)
                    
            # print("Next token:", tokenizer.decode(next_tokens.item()))
            # print("Next token score:", next_token_scores[0, next_tokens.item()].item())
                        
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs
            )
            
            if next_tokens.item() == tokenizer.eos_token_id:
                break
            
            if next_tokens.item() == caption_start_id and generate_image:
                cached_past_key_values = outputs.past_key_values
                caption_start_idx = input_ids.shape[-1]      
                 
            if next_tokens.item() == caption_end_id and generate_image:
                caption = tokenizer.decode(input_ids[0][caption_start_idx:-1]).strip()
                image = decode_with_sdxl(caption, sd_base, sd_refiner, generator)
                image_list.append(image)
                caption_list.append(caption)
                
                # [1, 3, 224, 224]
                input_images = image_processor(images=image, return_tensors='pt')['pixel_values']
                input_images = input_images.to(input_ids.device).to(torch.half if self.args.fp16 else torch.float)               

                # update past_key_values
                model_kwargs["past_key_values"] = cached_past_key_values     
            else:
                input_images = None               
            
        return {
            "sequences": input_ids,
            "image_list": image_list,
            "caption_list": caption_list
        }

def topp_logits_filter(scores, p):
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    # Keep at least 1 token
    sorted_indices_to_remove[..., -1 :] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, -float("Inf"))
    return scores
