import pandas as pd
import numpy as np
import re
import inspect
import torch
from utils import levenshtein, split_in_sentence
from IPython.display import clear_output
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, top_k_top_p_filtering
from transformers.generation import utils
#from transformers.generation.beam_search import BeamSearchScorer
from beam_search import BeamSearchScorer
from beam_search import _new_beam_search, _new_group_beam_search
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import ModelOutput
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    ClassifierFreeGuidanceLogitsProcessor,
)

def _prepare_attention_mask_for_generation(
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
) -> torch.LongTensor:
    is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
    is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

    # Check if input is input_ids and padded -> only then is attention_mask defined
    if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
        return inputs.ne(pad_token_id).long()
    else:
        return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)


def _prepare_encoder_decoder_kwargs_for_generation(
    model, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
) -> Dict[str, Any]:
    # 1. get encoder
    encoder = model.get_encoder()
    # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
    # as the inputs.
    if hasattr(encoder, "_hf_hook"):
        encoder._hf_hook.io_same_device = True

    # 2. Prepare encoder args and encoder kwargs from model kwargs.
    irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
    encoder_kwargs = {
        argument: value
        for argument, value in model_kwargs.items()
        if not any(argument.startswith(p) for p in irrelevant_prefix)
    }
    encoder_signature = set(inspect.signature(encoder.forward).parameters)
    encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
    if not encoder_accepts_wildcard:
        encoder_kwargs = {
            argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
        }

    # 3. make sure that encoder returns `ModelOutput`
    model_input_name = model_input_name if model_input_name is not None else model.main_input_name
    encoder_kwargs["return_dict"] = True
    encoder_kwargs[model_input_name] = inputs_tensor
    model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

    return model_kwargs


def _prepare_decoder_input_ids_for_generation(
        model,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
    """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
    # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
    # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
    if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
        decoder_input_ids = model_kwargs.pop("decoder_input_ids")
    elif "input_ids" in model_kwargs and model_input_name != "input_ids":
        decoder_input_ids = model_kwargs.pop("input_ids")
    else:
        decoder_input_ids = None

    # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
    decoder_start_token_id = model._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
    if device is None:
        device = model.device
    decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id

    # no user input -> use decoder_start_token_id as decoder_input_ids
    if decoder_input_ids is None:
        decoder_input_ids = decoder_input_ids_start
    # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
    elif model.config.model_type == "vision-encoder-decoder" and "donut" in model.name_or_path.lower():
        pass
    # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
    # decoder_attention_mask if provided)
    elif (decoder_input_ids[:, 0] != decoder_start_token_id).all().item():
        decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            decoder_attention_mask = torch.cat(
                (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                dim=-1,
            )
            model_kwargs["decoder_attention_mask"] = decoder_attention_mask

    return decoder_input_ids, model_kwargs


def _get_stopping_criteria(
    self, generation_config: GenerationConfig, stopping_criteria: Optional[StoppingCriteriaList]
) -> StoppingCriteriaList:
    criteria = StoppingCriteriaList()
    if generation_config.max_length is not None:
        max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
        criteria.append(
            MaxLengthCriteria(
                max_length=generation_config.max_length,
                max_position_embeddings=max_position_embeddings,
            )
        )
    if generation_config.max_time is not None:
        criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
    criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
    return criteria


def _expand_inputs_for_generation(
    expand_size: int = 1,
    is_encoder_decoder: bool = False,
    input_ids: Optional[torch.LongTensor] = None,
    **model_kwargs,
) -> Tuple[torch.LongTensor, Dict[str, Any]]:
    """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

    def _expand_dict_for_generation(dict_to_expand):
        for key in dict_to_expand:
            if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
        return dict_to_expand

    if input_ids is not None:
        input_ids = input_ids.repeat_interleave(expand_size, dim=0)

    model_kwargs = _expand_dict_for_generation(model_kwargs)

    if is_encoder_decoder:
        if model_kwargs.get("encoder_outputs") is None:
            raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
        model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

    return input_ids, model_kwargs


def _merge_criteria_processor_list(
    self,
    default_list: Union[LogitsProcessorList, StoppingCriteriaList],
    custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
) -> Union[LogitsProcessorList, StoppingCriteriaList]:
    if len(custom_list) == 0:
        return default_list
    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                raise ValueError(
                    f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                    f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                    " created by passing the corresponding arguments to generate or by the model's config default"
                    f" values. If you just want to change the default values of {object_type} consider passing"
                    f" them as arguments to `.generate()` instead of using a custom {object_type}."
                )
    default_list.extend(custom_list)
    return default_list


def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
) -> LogitsProcessorList:
    """
    This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
    instances used to modify the scores of the language model head.
    """
    # instantiate processors list
    processors = LogitsProcessorList()

    if generation_config.sequence_bias is not None:
        processors.append(SequenceBiasLogitsProcessor(sequence_bias=generation_config.sequence_bias))

    if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=generation_config.diversity_penalty,
                num_beams=generation_config.num_beams,
                num_beam_groups=generation_config.num_beam_groups,
            )
        )
    if (
            generation_config.encoder_repetition_penalty is not None
            and generation_config.encoder_repetition_penalty != 1.0
    ):
        processors.append(
            EncoderRepetitionPenaltyLogitsProcessor(
                penalty=generation_config.encoder_repetition_penalty, encoder_input_ids=encoder_input_ids
            )
        )
    if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
    if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
    if (
            generation_config.encoder_no_repeat_ngram_size is not None
            and generation_config.encoder_no_repeat_ngram_size > 0
    ):
        if self.config.is_encoder_decoder:
            processors.append(
                EncoderNoRepeatNGramLogitsProcessor(
                    generation_config.encoder_no_repeat_ngram_size, encoder_input_ids
                )
            )
        else:
            raise ValueError(
                "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
            )
    if generation_config.bad_words_ids is not None:
        processors.append(
            NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
        )
    if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
    ):
        processors.append(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
    if (
            generation_config.min_new_tokens is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_new_tokens > 0
    ):
        processors.append(
            MinNewTokensLengthLogitsProcessor(
                input_ids_seq_length, generation_config.min_new_tokens, generation_config.eos_token_id
            )
        )
    if prefix_allowed_tokens_fn is not None:
        processors.append(
            PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn, generation_config.num_beams // generation_config.num_beam_groups
            )
        )
    if generation_config.forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
    if generation_config.forced_eos_token_id is not None:
        processors.append(
            ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
        )
    if generation_config.remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    if generation_config.exponential_decay_length_penalty is not None:
        processors.append(
            ExponentialDecayLengthPenalty(
                generation_config.exponential_decay_length_penalty,
                generation_config.eos_token_id,
                input_ids_seq_length,
            )
        )
    if generation_config.suppress_tokens is not None:
        processors.append(SuppressTokensLogitsProcessor(generation_config.suppress_tokens))
    if generation_config.begin_suppress_tokens is not None:
        begin_index = input_ids_seq_length
        begin_index = (
            begin_index
            if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
            else begin_index + 1
        )
        if generation_config.forced_decoder_ids is not None:
            # generation starts after the last token that is forced
            begin_index += generation_config.forced_decoder_ids[-1][0]
        processors.append(
            SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
        )
    if generation_config.forced_decoder_ids is not None:
        processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids))
    if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
        processors.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
    processors = _merge_criteria_processor_list(self, processors, logits_processor)
    # `LogitNormalization` should always be the last logit processor, when present
    if generation_config.renormalize_logits is True:
        processors.append(LogitNormalization())
    return processors

def generate_with_new_beam_search(
        model,
        inputs_ids,
        num_beams,
        device,
        num_beam_groups: Optional[int] = 1,
        max_length: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = [],
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        **kwargs,
):

    if max_length is None:
        max_length = inputs_ids.shape[-1]

    if generation_config is None:
        # legacy: users may modify the model configuration to control generation -- update the generation config
        # model attribute accordingly, if it was created from the model config

        generation_config = model.generation_config

    model_kwargs = generation_config.update(**kwargs)

    batch_size = inputs_ids.shape[0]

    model_input_name = 'input_ids'

    model_kwargs["output_attentions"] = False
    model_kwargs["output_hidden_states"] = False
    model_kwargs["use_cache"] = True
    model_kwargs["attention_mask"] = _prepare_attention_mask_for_generation(
        inputs_ids, 1, 2
    )
    model_kwargs = _prepare_encoder_decoder_kwargs_for_generation(
        model, inputs_ids, model_kwargs, model_input_name
    )

    input_ids, model_kwargs = _prepare_decoder_input_ids_for_generation(
        model,
        batch_size=batch_size,
        model_input_name=model_input_name,
        model_kwargs=model_kwargs,
        decoder_start_token_id=2,
        bos_token_id=0,
        device=device,
    )

    input_ids_seq_length = input_ids.shape[-1]
    has_default_max_length = False

    if num_beam_groups > 1:
        generation_config.num_beams = num_beams
        generation_config.num_beam_groups = num_beam_groups

    logits_processor = _get_logits_processor(
        model,
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=LogitsProcessorList(),
    )

    stopping_criteria = _get_stopping_criteria(
        model, generation_config=model.generation_config, stopping_criteria=StoppingCriteriaList()
    )

    beam_search_mode = (
            (num_beams > 1)
            and (num_beam_groups == 1)
    )

    group_beam_search_mode = (
            (num_beams > 1)
            and (num_beam_groups > 1)
    )

    #start beam_search
    if beam_search_mode:
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            length_penalty=1.0,
            do_early_stopping=False,
            num_beam_hyps_to_keep=1,
            max_length=max_length,
        )

        input_ids, model_kwargs = _expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=num_beams,
            is_encoder_decoder=True,
            **model_kwargs,
        )

        output = _new_beam_search(
            model,
            input_ids,
            beam_scorer,
            inputs=inputs_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=1,
            eos_token_id=2,
            output_scores=True,
            return_dict_in_generate=True,
            synced_gpus=False,
            **model_kwargs,
        )
    elif group_beam_search_mode:
        if num_beams % num_beam_groups != 0:
            raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")
        if generation_config.diversity_penalty == 0.0:
            raise ValueError(
                "`diversity_penalty` should be greater than `0.0`, otherwise your beam groups will be identical."
            )
        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            length_penalty=1.0,
            do_early_stopping=False,
            num_beam_hyps_to_keep=1,
            num_beam_groups=num_beam_groups,
            max_length=max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = _expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=num_beams,
            is_encoder_decoder=True,
            **model_kwargs,
        )
        # 13. run beam search
        return _new_group_beam_search(
            model,
            input_ids,
            beam_scorer,
            inputs=inputs_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=1,
            eos_token_id=2,
            output_scores=True,
            return_dict_in_generate=True,
            synced_gpus=False,
            **model_kwargs,
        )

    else:
        raise("No mode selscted")


    return output


def process_with_new_beam_search(model, tokenizer, input_text, beam_width, device, alpha = 0.9):
    language_list = get_tokens_as_list(list(tokenizer.lang_code_to_id.keys()))
    language_list.remove([tokenizer.lang_code_to_id["ita_Latn"]])
    inputs = tokenizer(input_text.strip(), return_tensors="pt").to(device)
    inputs_ids = inputs['input_ids']
    max_length = round(len(inputs_ids[0]) * 1)  # previous coefficient *1.25
    output = generate_with_new_beam_search(
        model,
        inputs_ids,
        beam_width,
        device,
        max_length=max_length,
        bad_words_ids=language_list,
        forced_bos_token_id=tokenizer.lang_code_to_id["ita_Latn"],
        alpha = alpha,
    )
    #print(output.scores)
    #print(tokenizer.decode(output.sequences[0]))
    return tokenizer.decode(output.sequences[0], skip_special_tokens=True)


def get_tokens_as_list(word_list):
    model_name = "facebook/nllb-200-distilled-600M" #TO FIX: put model_name in class (or somthing else)
    tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list


def main():
    utils.beam_search = _new_beam_search
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="ita_Latn", tgt_lang="ita_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    ocr_text = "MORINI, Segretmio, legge: (C Per gli interventi di competenza del Ministero dei lavori pubblici previsti dalla legge 14 marzo 1968, n. 292, è autorizzata la spesa di lire 3.000 milioni. L’autorizzazione di spesa di cui all’articolo 35, quarto comma del decreto-legge 13 maggio 1976, n. 227, convertito nella legge 29 maggio 1976, n. 336, aumentata con l’articolo 35 del decreto-legge 18 settembre 1976, n. 648, convertito nella legge 30 ottobre 1976, n. 730, B ulteriormente ‘aumentata di lire 10.000 milioni. I lavori di ricostruzione e ’di riparazione degli edifici di culto, dopo l’approvazione del progetto esecutivo, possono essere affidati in concessione all’ordinario diocesano competente per territorio. In ogni progetto è computata, per spese .di compilazione, direzione e sorveglianza, da corrispondersi all’ordinario diocesano, una somma corrispondente al 5 per cento dell’ammontare dei lavori eseguiti. I1 collaudo delle opere è effettuato a cura dello Stato n. (B approvato)."

    splitted_text = split_in_sentence(ocr_text)
    translated_text = []
    beam_width = 5
    alpha = 0.9
    for text in splitted_text:
        txt = process_with_new_beam_search(model, tokenizer, text, beam_width, device, alpha) #beam_search
        translated_text.append(txt)
        print("----------")
        print(text)
        print(txt)

    print("===================")
    print(ocr_text)
    print("")
    print(' '.join(translated_text))

    return None

if __name__ == "__main__":
    main()