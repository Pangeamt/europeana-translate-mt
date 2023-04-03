from pangeamt_nlp.processor.base.validator_base import ValidatorBase
from pangeamt_nlp.seg import Seg


class ContainsTestVal(ValidatorBase):
    NAME = "contains_test_val"

    DESCRIPTION_TRAINING = """
            Remove pairs if the pair is in the test file.
            Parameters: path test file(str)
        """

    DESCRIPTION_DECODING = """
            Validators do not apply to decoding.
        """

    def __init__(self, src_lang: str, tgt_lang: str, file: str) -> None:
        super().__init__(src_lang, tgt_lang)
        self.initialized = False
        self.file = file

    def validate(self, seg: Seg) -> bool:
        if not self.initialized:
            self.initialize()
        if seg.src.strip('\n') in self.test_lines:
                    return False
        return True

    def initialize(self) -> None:
        open_file = open(self.file, 'r', encoding='utf-8')
        self.test_lines = set()
        for line in open_file:
            if line == '\n':
                continue
            self.test_lines.add(line.strip('\n').strip(' '))
        open_file.close()
        self.initialized = True
