import numpy as np
import torch
from transformers import AutoTokenizer
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections import UserDict

import torch
import torch.distributed as dist
from torch import nn

#from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from transformers.utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
#from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
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
    #UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
from abc import ABC, abstractmethod


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for [`~PreTrainedModel.beam_search`] and
    [`~PreTrainedModel.beam_sample`].
    """

    @abstractmethod
    #@add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    #@add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        max_length: int,
        **kwargs,
    ) -> torch.LongTensor:
        raise NotImplementedError("This is an abstract method.")


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = None):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

        if not isinstance(self.early_stopping, bool) and self.max_length is None:
            raise ValueError(
                "When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the"
                " BeamScorer class instance at initialization time."
            )

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float, beam_indices: Optional[torch.LongTensor] = None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False

        # `True`: stop as soon as at least `num_beams` hypotheses are finished
        if self.early_stopping is True:
            return True
        # `False`: heuristic -- compute best possible score from `cur_len`, even though it is not entirely accurate
        #  when `length_penalty` is positive. See the discussion below for more details.
        # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
        elif self.early_stopping is False:
            highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret
        # `"never"`: compute the best possible score, depending on the signal of `length_penalty`
        else:
            # `length_penalty` > 0.0 -> max denominator is obtaned from `max_length`, not from `cur_len` -> min
            # abs(`highest_attainable_score`) is obtained -> `highest_attainable_score` is negative, hence we obtain
            # its max this way
            if self.length_penalty > 0.0:
                highest_attainable_score = best_sum_logprobs / self.max_length**self.length_penalty
            # the opposite logic applies here (max `highest_attainable_score` from `cur_len`)
            else:
                highest_attainable_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= highest_attainable_score
            return ret


class BeamSearchScorer(BeamScorer):
    r"""
        [`BeamScorer`] implementing standard beam search decoding.

        Adapted in part from [Facebook's XLM beam search
        code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

        Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
        implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

        Args:
            batch_size (`int`):
                Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
            num_beams (`int`):
                Number of beams for beam search.
            device (`torch.device`):
                Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
                allocated.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
                the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
                likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
                `length_penalty` < 0.0 encourages shorter sequences.
            do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
                Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
                `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
                heuristic is applied and the generation stops when is it very unlikely to find better candidates;
                `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
                beam search algorithm).
            num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
                The number of beam hypotheses that shall be returned upon calling
                [`~transformer.BeamSearchScorer.finalize`].
            num_beam_groups (`int`):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
                See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            max_length (`int`, *optional*):
                The maximum length of the sequence to be generated.
        """

    def __init__(
            self,
            batch_size: int,
            num_beams: int,
            device: torch.device,
            length_penalty: Optional[float] = 1.0,
            do_early_stopping: Optional[Union[bool, str]] = False,
            num_beam_hyps_to_keep: Optional[int] = 1,
            num_beam_groups: Optional[int] = 1,
            max_length: Optional[int] = None,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        # self._beam_hyps[i*self.num_beam_groups+j] is the beam_hyps of the j-th group in the i-th mini-batch.
        # If group_beam_search is not used, the list consists of `batch_size` beam_hyps.
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.group_size,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
                max_length=max_length,
            )
            for _ in range(batch_size * self.num_beam_groups)
        ]
        # self._done[i*self.num_beam_groups+j] indicates whether the generation of the beam_hyps of the j-th group
        # in the i-th mini-batch is complete.
        self._done = torch.tensor(
            [False for _ in range(batch_size * self.num_beam_groups)], dtype=torch.bool, device=self.device
        )

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1,"
                " one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be"
                f" divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            beam_indices: Optional[torch.LongTensor] = None,
            group_index: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        cur_len = input_ids.shape[-1] + 1  # add up to the length which the next_scores is calculated on
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if not (batch_size == (input_ids.shape[0] // self.group_size)):
            if self.num_beam_groups > 1:
                raise ValueError(
                    f"A group beam size of {input_ids.shape[0]} is used as the input, but a group beam "
                    f"size of {self.group_size} is expected by the beam scorer."
                )
            else:
                raise ValueError(
                    f"A beam size of {input_ids.shape[0]} is used as the input, but a beam size of "
                    f"{self.group_size} is expected by the beam scorer."
                )

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx in range(batch_size):
            batch_group_idx = batch_idx * self.num_beam_groups + group_index
            if self._done[batch_group_idx]:
                if self.num_beams < len(self._beam_hyps[batch_group_idx]):
                    raise ValueError(f"Batch can only be done if at least {self.num_beams} beams have been generated")
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError("Generated beams >= num_beams -> eos_token_id and pad_token have to be defined")
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None

                    self._beam_hyps[batch_group_idx].add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        beam_indices=beam_index,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id:"
                    f" {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_group_idx] = self._done[batch_group_idx] or self._beam_hyps[batch_group_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
            self,
            input_ids: torch.LongTensor,
            final_beam_scores: torch.FloatTensor,
            final_beam_tokens: torch.LongTensor,
            final_beam_indices: torch.LongTensor,
            max_length: int,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            beam_indices: Optional[torch.LongTensor] = None,
            original = None,
            tokenizer = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps) // self.num_beam_groups

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_group_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_group_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for index_per_group in range(self.group_size):
                batch_beam_idx = batch_group_idx * self.group_size + index_per_group
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                beam_hyp.add(final_tokens, final_score, beam_indices=beam_index)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i in range(batch_size):
            beam_hyps_in_batch = self._beam_hyps[i * self.num_beam_groups: (i + 1) * self.num_beam_groups]
            candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]
            # TODO add liv. distance here for each candidate - Done
            if original is not None:
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(self.name_or_path, src_lang="ita_Latn", tgt_lang = "ita_Latn")
                for idx, candidate in enumerate(candidate_beams):
                    candidate = list(candidate)
                    s1 = tokenizer.decode(candidate[1])
                    candidate[0] = levenshtein(s1,original[0])
                    candidate_beams[idx] = tuple(candidate)
            sorted_hyps = sorted(candidate_beams, key=lambda x: x[0], reverse=True)
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append hyp to lists
                best.append(best_hyp)

                # append indices to list
                best_indices.append(best_index)

                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)

        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        else:
            indices = None

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            if pad_token_id is None:
                raise ValueError("`pad_token_id` has to be defined")
            decoded.fill_(pad_token_id)

        if indices is not None:
            indices.fill_(-1)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[i, : sent_lengths[i]] = hypo

            if indices is not None:
                indices[i, : len(best_idx)] = torch.tensor(best_idx)

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "beam_indices": indices,
            }
        )


def _new_beam_search(self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    inputs: Optional[torch.Tensor] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    alpha: Optional[float] = 0.9,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

        alpha (`float`, *optional*, defaults to 0.9):
            Scoring parameter with levenshtein distance

        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    """
    # init values

    #print("HELLO WORLD, THIS IS A MODIFIED VERSION OF BEAM_SEARCH!")
    tokenizer = AutoTokenizer.from_pretrained(self.name_or_path, src_lang="ita_Latn", tgt_lang="ita_Latn")
    language_list = get_tokens_as_list(list(tokenizer.lang_code_to_id.keys()))
    language_list.remove([tokenizer.lang_code_to_id["ita_Latn"]])

    s2 = tokenizer.batch_decode(inputs)  # string representation of input for edit distance calculation

    word_tokens = {x: tokenizer.encode(x, add_special_tokens=False) for x in s2[0].split()}  # list of tokens for each word
    word_tokens = list(word_tokens.values())
    word_index = 0  # index of word from sentences
    word_token_index = 0  # index of generated token


    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
            next_token_scores_processed
        )

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
        n_eos_tokens = len(eos_token_id) if eos_token_id else 0

        #MODIFICATION OF ORIGINAL FUNCTION
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, max(5, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True   # change max(2... -> max(5...
        )
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        check_inp = []
        different_tokens = []
        for idx in range(next_token_scores.shape[-1]):
        #for (next_score, next_token, next_index) in zip(next_token_scores[0,idx], next_tokens[0,idx], next_indices[0,idx]):
            next_token = next_tokens[0, idx]
            next_index = next_indices[0, idx]

            # whole string distance
            '''
            s1_tokens = inputs.clone().detach()
            if input_ids.shape[-1] > 1 and input_ids.shape[-1] <= len(inputs[0]):
                s1_tokens[0,0:(input_ids.shape[-1]-1)] = input_ids[next_index, 1:]
            elif input_ids.shape[-1] > len(inputs[0]):
                s1_tokens = torch.unsqueeze(input_ids[next_index, 1:], 0).clone().detach()
            if input_ids.shape[-1] < len(inputs[0]):
                s1_tokens[0, input_ids.shape[-1] - 1] = next_token
            else:
                s1_tokens = torch.cat((s1_tokens, torch.tensor([[next_token]]).to(input_ids.device)), dim=1)
            s1 = tokenizer.batch_decode(s1_tokens)
            #distance = levenshtein(s1[0], s2[0])

            # substring (word) distance -- much worse

            if word_index < len(word_tokens):
                s2_tokens = word_tokens[word_index]
            else:
                s2_tokens = tokenizer.encode(" ")
            s1_word_tokens = s2_tokens.copy()
            
            if word_token_index > 0:
                s1_word_tokens[:word_token_index] = input_ids[next_index, -word_token_index:]
                s1_word_tokens[word_token_index] = next_token
            else:
                s1_word_tokens[0] = next_token
            '''

            s1_tokens = inputs.clone().detach()
            if input_ids.shape[-1] > 1 and input_ids.shape[-1] <= len(inputs[0]):
                s1_tokens[0, 0:(input_ids.shape[-1] - 1)] = input_ids[next_index, 1:]
            elif input_ids.shape[-1] > len(inputs[0]):
                s1_tokens = torch.unsqueeze(input_ids[next_index, 1:], 0).clone().detach()
            if input_ids.shape[-1] < len(inputs[0]):
                s1_tokens[0, input_ids.shape[-1] - 1] = next_token
            else:
                s1_tokens = torch.cat((s1_tokens, torch.tensor([[next_token]]).to(input_ids.device)), dim=1)
            s1 = tokenizer.batch_decode(s1_tokens)
            distance = levenshtein(s1[0], s2[0])

            # substring (word) distance -- much worse
            '''
            if word_index == 0:
                s2_tokens = word_tokens[word_index:word_index + 2]
                s2_tokens = [item for sublist in s2_tokens for item in sublist]
            elif word_index < len(word_tokens) - 1 and word_index > 0:
                s2_tokens = word_tokens[word_index - 1:word_index + 2]
                s2_tokens = [item for sublist in s2_tokens for item in sublist]
            elif word_index < len(word_tokens):
                s2_tokens = word_tokens[word_index - 1:word_index + 1]
                s2_tokens = [item for sublist in s2_tokens for item in sublist]
            else:
                s2_tokens = tokenizer.encode(" ")
            # TODO change s1 to 3-words (i-1,i,i+1)
            s1_word_tokens = s2_tokens.copy()

            if word_index > 0:
                start_change = len(word_tokens[word_index - 1]) + word_token_index
            else:
                start_change = word_token_index
            if start_change > 0:
                s1_word_tokens[:start_change] = input_ids[next_index, -start_change:]
            s1_word_tokens[start_change] = next_token
            s1_word = tokenizer.decode(s1_word_tokens)
            s2_word = tokenizer.decode(s2_tokens)
            distance = levenshtein(s1_word, s2_word)
            '''



            next_score = score(s1_tokens, self)
            if next_score < 0:
                #next_token_scores[0, idx] = next_token_scores[0, idx] / (0.9**distance)  # 0.8 -> 1
                next_token_scores[0, idx] = next_score / (alpha ** distance)  # 0.8 -> 1
            else:
                next_token_scores[0, idx] = next_score * (alpha ** distance)  # 0.8 -> 1

            check_token = torch.cat((input_ids[next_index], torch.tensor([next_token]).to(input_ids.device)), dim=0).tolist()
            ct_exist = False
            for l in check_inp:
                if l == check_token:
                    ct_exist = True
            if not ct_exist:
                check_inp.append(check_token)
                different_tokens.append({
                    'token': next_token,
                    'index': next_index,
                    'score': next_token_scores[0, idx],
                    'distance': distance,
                    'sent_score': next_score
                })

            # substring (word) distance
            '''
            word_tokens = {x: tokenizer.encode(x, add_special_tokens=False) for x 
                            in s2.split()}  # list of tokens for each word
            word_index = 0  # index of word from sentences
            word_token_index = 0  # index of generated token
            start_word_token_index = 0  # index of first token of generated word
            '''

        new_tokens, new_indexes, new_scores, _, _ = zip(*map(dict.values, different_tokens))
        new_scores = torch.unsqueeze(torch.stack(new_scores, dim=0), 0)
        new_tokens = torch.unsqueeze(torch.stack(new_tokens, dim=0), 0)
        new_indexes = torch.unsqueeze(torch.stack(new_indexes, dim=0), 0)
        new_scores, indeces = torch.sort(new_scores, descending=True)
        new_tokens = new_tokens[0, indeces]
        new_indexes = new_indexes[0, indeces]
        if new_scores.shape[-1] < batch_beam_size:
            new_scores = new_scores.repeat(1, batch_beam_size)[:,0:batch_beam_size]
            new_tokens = new_tokens.repeat(1, batch_beam_size)[:,0:batch_beam_size]
            new_indexes = new_indexes.repeat(1, batch_beam_size)[:,0:batch_beam_size]

        next_token_scores = new_scores
        next_tokens = new_tokens
        next_indices = new_indexes

        # index update for substring (word) distance
        if word_index < len(word_tokens):
            word_token_index += 1
            if len(word_tokens[word_index]) == word_token_index:
                word_token_index = 0
                word_index += 1
        else:
            word_token_index = 0

        #if word_index == 28 or word_index == 2:
        #    print('stop here')

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

        # remove too different branch and replace with best
        best_input = None
        best_distance = np.inf
        for idx, s in enumerate(input_ids):
            s1_tokens = inputs.clone().detach()
            if input_ids.shape[-1] > 1 and input_ids.shape[-1] <= len(inputs[0]):
                s1_tokens[0, 0:(input_ids.shape[-1] - 1)] = input_ids[idx, 1:]
            elif input_ids.shape[-1] > len(inputs[0]):
                s1_tokens = torch.unsqueeze(input_ids[idx, 1:], 0).clone().detach()
            s1 = tokenizer.batch_decode(s1_tokens)
            distance = levenshtein(s1[0], s2[0])
            if distance < best_distance:
                best_input = input_ids[idx]
                best_distance = distance
            if distance > len(s) / 2:
                input_ids[idx] = best_input


    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        original=s2,
        tokenizer=tokenizer,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]


# TODO add score to distance in main loop of beam_search
def score(sentence_tokens, model):
    #tokenize_input = tokenizer.tokenize(sentence)
    #tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(model.device)
    loss = model(sentence_tokens, labels=sentence_tokens)
    return -loss[0].item()


def levenshtein(s1, s2):
    """
    calculate levenshtein distance between strings s1 and s2
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_tokens_as_list(word_list):
    model_name = "facebook/nllb-200-distilled-600M" #TO FIX: put model_name in class (or somthing else)
    tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list
