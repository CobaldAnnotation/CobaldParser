from typing import override

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import Pipeline
from transformers.modeling_outputs import ModelOutput

from src.lemmatize_helper import reconstruct_lemma


# Download punkt tokenizer
nltk.download('punkt_tab')


class ConlluTokenClassificationPipeline(Pipeline):
    def __init__(self, model, language, **kwargs):
        super().__init__(model=model, framework='pt', **kwargs)
        self.language = language

    @override
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    @override
    def preprocess(self, inputs: str) -> dict:
        if not isinstance(inputs, str):
            raise ValueError("pipeline input must be string (text)")
        
        sentences = [sentence for sentence in sent_tokenize(inputs, self.language)]
        # stash for later post‐processing
        self._texts = sentences

        words = [
            [word for word in word_tokenize(sentence, preserve_line=True)]
            for sentence in sentences
        ]
        model_inputs = {"words": words}
        return model_inputs
    
    @override
    def _forward(self, model_inputs: dict) -> ModelOutput:
        model_outputs = self.model(**model_inputs, inference_mode=True)
        return model_outputs
    
    @override
    def postprocess(self, model_outputs: ModelOutput) -> list[dict]:

        n_sentences = len(model_outputs.words)
        
        sentences = []
        for i in range(n_sentences):
        
            def select_arcs(arcs, batch_idx):
                # Select arcs where batch index == batch_idx
                # Return tensor of shape [n_selected_arcs, 3]
                return arcs[arcs[:, 0] == batch_idx][:, 1:]
            
            # Model outputs are padded tensors, so only leave first `n_words` labels.
            n_words = len(model_outputs.words[i])

            sentence = self._postprocess_sentence(
                text=self._texts[i],
                words=model_outputs.words[i],
                lemma_rule_ids=model_outputs.lemma_rules[i, :n_words].tolist(),
                morph_feats_ids=model_outputs.morph_feats[i, :n_words].tolist(),
                deps_ud=select_arcs(model_outputs.deps_ud, i).tolist(),
                deps_eud=select_arcs(model_outputs.deps_eud, i).tolist(),
                misc_ids=model_outputs.miscs[i, :n_words].tolist(),
                deepslot_ids=model_outputs.deepslots[i, :n_words].tolist(),
                semclass_ids=model_outputs.semclasses[i, :n_words].tolist()
            )
            sentences.append(sentence)
        return sentences

    def _postprocess_sentence(
        self,
        text: str,
        words: list[str],
        lemma_rule_ids: list[int],
        morph_feats_ids: list[int],
        deps_ud: list[list[int]],
        deps_eud: list[list[int]],
        misc_ids: list[int],
        deepslot_ids: list[int],
        semclass_ids: list[int]
    ) -> dict:

        ids = self._enumerate_words(words)

        lemmas = [
            reconstruct_lemma(
                word,
                self.model.config.id2lemma_rule[lemma_rule_id]
            )
            for word, lemma_rule_id in zip(words, lemma_rule_ids, strict=True)
        ]

        upos, xpos, feats = zip(
            *[
                self.model.config.id2morph_feats[morph_feats_id].split('#')
                for morph_feats_id in morph_feats_ids
            ],
            strict=True
        )
        upos, xpos, feats = list(upos), list(xpos), list(feats)

        # ids stores inverse mapping from internal numeration to the standard one,
        # so simply use ids[internal_idx] to get conllu index.
        convert_arcs_to_conllu_format = lambda arcs, id2rel: [
            (
                ids[arc_from],
                ids[arc_to] if arc_from != arc_to else '0',
                id2rel[deprel_id]
            )
            for arc_from, arc_to, deprel_id in arcs
        ]
        deps_ud = convert_arcs_to_conllu_format(deps_ud, self.model.config.id2rel_ud)
        deps_eud = convert_arcs_to_conllu_format(deps_eud, self.model.config.id2rel_eud)

        miscs = [self.model.config.id2misc[misc_id] for misc_id in misc_ids]
        deepslots = [self.model.config.id2deepslot[deepslot_id] for deepslot_id in deepslot_ids]
        semclasses = [self.model.config.id2semclass[semclass_id] for semclass_id in semclass_ids]

        sentence = {
            "text": text,
            "ids": ids,
            "words": words,
            "lemmas": lemmas,
            "upos": upos,
            "xpos": xpos,
            "feats": feats,
            "deps_ud": deps_ud,
            "deps_eud": deps_eud,
            "miscs": miscs,
            "deepslots": deepslots,
            "semclasses": semclasses
        }
        return sentence
    
    @staticmethod
    def _enumerate_words(words: list[str]) -> list[str]:
        ids = []
        current_id = 0
        current_null_count = 0
        for word in words:
            if word == "#NULL":
                current_null_count += 1
                ids.append(f"{current_id}.{current_null_count}")
            else:
                current_id += 1
                current_null_count = 0
                ids.append(f"{current_id}")
        return ids