#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import sentencepiece as spm


class SentencePieceSegmenter:
    def __init__(self, model_path: str):
        self._segmenter = spm.SentencePieceProcessor(model_file=model_path)

    def apply(self, text: str) -> str:
        return (" ").join(self._segmenter.encode(text, out_type=str)) + "\n"

    def undo(self, text: str) -> str:
        return self._segmenter.decode(text.split(" "))

    @staticmethod
    def learn(
            input: str,
            output: str,
            model_type: str = "unigram",
            vocab_size: int = 8000,
            symbols=[]
    ):
        spm.SentencePieceTrainer.train(
            input=input,
            model_prefix=output,
            model_type=model_type,
            vocab_size=vocab_size,
            user_defined_symbols=symbols,
            train_extremely_large_corpus=True
        )
