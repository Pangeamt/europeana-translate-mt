from pangeamt_nlp.processor.base.validator_base import ValidatorBase
from pangeamt_nlp.seg import Seg


class EqualsSentencesVal(ValidatorBase):
    NAME = "equals_sentences_val"

    DESCRIPTION_TRAINING = """
        Checks if src and target lines are equal, in that case skips this seg.
    """

    DESCRIPTION_DECODING = """
        Validators do not apply to decoding.
    """

    def __init__(self, src_lang: str, tgt_lang: str) -> None:
        super().__init__(src_lang, tgt_lang)
        self.MISSING_SEG_TOKEN = "[[[@@@MISSING_TRANSLATION@@@]]]"

    def validate(self, seg: Seg) -> bool:
        if len(seg.src.split(' ')) >= 5 or seg.src == self.MISSING_SEG_TOKEN:
            if seg.src.lower() == seg.tgt.lower():
                return False

        return True
