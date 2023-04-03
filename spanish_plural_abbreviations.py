from pangeamt_nlp.processor.base.normalizer_base import NormalizerBase
from pangeamt_nlp.seg import Seg
import re
import numpy as np


class SpanishPluralAbbreviations(NormalizerBase):
    DESCRIPTION_TRAINING = """
            Normalizes the plural abbreviations in Spanish following its rules.
        """

    DESCRIPTION_DECODING = """
            
        """

    NAME = "spanish_plural_abbreviations"


    SPACES = ["\u0020", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009",
          "\u200A", "\u202F", "\u205F", "\u3000"]

    def __init__(self, src_lang: str, tgt_lang: str):
        super().__init__(src_lang, tgt_lang)

    def _normalize(self, text):
        iterator = re.finditer(r'([A-Z])\1[\.\s]{0,2}([A-Z]){2}\.?', text)
        res_text = []
        for match in iterator:
            plural_abbreviation = match.group(0)
            final_index = text.index(plural_abbreviation) + len(plural_abbreviation)
            part_of_the_res_text = re.sub(r'([A-Z])\1[\.\s]{0,2}[A-Z]{2}\.?',
                         match.group(1) * 2 + ".\u00A0" + match.group(2) * 2 + ".", text[:final_index])
            res_text.append(part_of_the_res_text)
            text = text[final_index:]
        res_text.append(text)
        return "".join(res_text)

    # Called when training
    def process_train(self, seg: Seg) -> None:
        if self.get_src_lang() == "es":
            seg.src = self._normalize(seg.src)
        if seg.tgt is not None and self.get_tgt_lang() == "es":
            seg.tgt = self._normalize(seg.tgt)

    # Called when using model (before calling model to translate)
    def process_src_decoding(self, seg: Seg) -> None:
        if self.get_src_lang() == "es":
            seg.src = self._normalize(seg.src)

    # Called after the model translated (in case this would be necessary; usually not the case)
    def process_tgt_decoding(self, seg: Seg) -> None:
        if self.get_tgt_lang() == "es":
            seg.tgt = self._normalize(seg.tgt)
