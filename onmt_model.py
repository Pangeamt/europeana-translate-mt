import torch
from pangeamt_nlp.types import Translation, Pred_sent
from pangeamt_nlp.translation_model.translation_model_base import (
    TranslationModelBase,
)
from onmt.translate import GNMTGlobalScorer
from onmt.translate import Translator, TranslationBuilder
from onmt.bin.train import train as onmt_train
from onmt.bin.train import _get_parser
import onmt.inputters as inputters
from onmt.inputters.text_dataset import TextDataReader
from onmt.utils.parse import ArgumentParser
from onmt.model_builder import build_base_model
from onmt.utils.misc import set_random_seed
from onmt.utils import Optimizer
from onmt.trainer import build_trainer
from typing import List, Tuple, Dict


class ONMT_model(TranslationModelBase):
    NAME = "onmt"
    INITIALIZED = False

    DEFAULT = (
        "-layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 "
        "-heads 8 -encoder_type transformer -decoder_type transformer "
        "-position_encoding -train_steps 500000 -max_generator_batches 2 "
        "-dropout 0.1 -batch_size 4096 -batch_type tokens -normalization "
        "tokens -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method "
        "noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 "
        "-param_init 0 -param_init_glorot -label_smoothing 0.1 -valid_steps "
        "10000 -save_checkpoint_steps 10000 -keep_checkpoint 10 "
        "-early_stopping 8"
    )

    DEFAULT_DECODING = {
        "gpu": -1, "n_best": 5, "min_length": 0, "max_length": 300,
        "ratio": 0.0, "beam_size": 5, "random_sampling_topk": 1,
        "random_sampling_temp": 1, "stepwise_penalty": None,
        "dump_beam": False, "block_ngram_repeat": 0, "replace_unk": True,
        "phrase_table": '', "data_type": 'text', "verbose": False,
        "report_time": False, "copy_attn": False, "out_file": None,
        "report_align": False, "report_score": True, "logger": None,
        "seed": 829
    }

    def __init__(self, path: str, **kwargs) -> None:
        super().__init__()
        if kwargs == {}:
            kwargs = self.DEFAULT_DECODING
        kwargs["ignore_when_blocking"] = frozenset({})
        kwargs["global_scorer"] = GNMTGlobalScorer(0.0, -0.0, "none", "none")
        self._args = kwargs
        self._load(path, **kwargs)
        self._trainer = None

    @staticmethod
    def train(
        data_path: str, model_path: str, *args, gpu: str = None, **kwargs
    ):

        prepend_args = (
            f"-data {data_path}/data -save_model {model_path}/model "
        )
        if gpu:
            apend_args = f"-gpu_ranks {gpu} -log_file {data_path}/log.txt"
        else:
            apend_args = f"-log_file {data_path}/log.txt"

        args = (
            prepend_args + (" ").join(list(args)) + " " + apend_args
        ).split(" ")

        parser = _get_parser()

        opt = parser.parse_args(list(args))
        onmt_train(opt)

    def _load(self, path: str, **kwargs) -> None:
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage
        )
        self._model_opts = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(self._model_opts)
        ArgumentParser.validate_model_opts(self._model_opts)

        # Extract vocabulary
        vocab = checkpoint["vocab"]
        if inputters.old_style_vocab(vocab):
            self._fields = inputters.load_old_vocab(
                vocab, "text", dynamic_dict=False
            )
        else:
            self._fields = vocab

        # Train_steps
        self._train_steps = self._model_opts.train_steps

        # Build openmmt model
        def _use_gpu(**kwargs):
            return kwargs["gpu"] >= 0

        self._opennmt_model = build_base_model(
            self._model_opts,
            self._fields,
            _use_gpu(**kwargs),
            checkpoint,
            kwargs["gpu"],
        )

        self._translator = Translator(
            self._opennmt_model,
            self._fields,
            TextDataReader(),
            TextDataReader(),
            **kwargs
        )

    def setup_online_trainer(self, online_learning: Dict = None):
        # Optim
        optimizer_opt = type("", (), {})()
        optimizer_opt.optim = "sgd"
        optimizer_opt.learning_rate = 0.05
        optimizer_opt.train_from = ""
        optimizer_opt.adam_beta1 = 0
        optimizer_opt.adam_beta2 = 0
        optimizer_opt.model_dtype = "fp32"
        optimizer_opt.decay_method = "none"
        optimizer_opt.start_decay_steps = 100000
        optimizer_opt.learning_rate_decay = 1.0
        optimizer_opt.decay_steps = 100000
        optimizer_opt.max_grad_norm = 5

        trainer_opt = type("", (), {})()
        trainer_opt.lambda_coverage = 0.0
        trainer_opt.lambda_align = 0.0
        trainer_opt.src_noise = []
        trainer_opt.copy_attn = False
        trainer_opt.label_smoothing = 0.0
        trainer_opt.truncated_decoder = 0
        trainer_opt.model_dtype = "fp32"
        trainer_opt.max_generator_batches = 32
        trainer_opt.normalization = "sents"
        trainer_opt.accum_count = [1]
        trainer_opt.accum_steps = [0]
        trainer_opt.world_size = 1
        trainer_opt.average_decay = 0
        trainer_opt.average_every = 1
        trainer_opt.dropout = 0
        trainer_opt.dropout_steps = (0,)
        trainer_opt.gpu_verbose_level = 0
        trainer_opt.early_stopping = 0
        trainer_opt.early_stopping_criteria = (None,)
        trainer_opt.tensorboard = False
        trainer_opt.report_every = 50
        trainer_opt.gpu_ranks = []
        if self._args["gpu"] != -1:
            trainer_opt.gpu_ranks = [self._args["gpu"]]

        if online_learning is not None:
            for key, value in online_learning.items():
                if key in optimizer_opt.__dict__:
                    optimizer_opt.__dict__[key] = value
                if key in trainer_opt.__dict__:
                    trainer_opt.__dict__[key] = value

        self._optim = Optimizer.from_opt(
            self._opennmt_model, optimizer_opt, checkpoint=None
        )

        self._trainer = build_trainer(
            trainer_opt,
            self._args["gpu"],
            self._opennmt_model,
            self._fields,
            self._optim,
        )

    def translate(self, srcs, n_best=1):
        # Set model mode to eval
        self._opennmt_model.eval()
        self._opennmt_model.generator.eval()

        dataset = inputters.Dataset(
            self._fields,
            readers=([self._translator.src_reader]),
            data=[("src", srcs)],
            dirs=[None],
            sort_key=inputters.str2sortkey[self._translator.data_type],
            filter_pred=self._translator._filter_pred,
        )

        data_iter = inputters.OrderedIterator(
            dataset=dataset,
            device=self._translator._dev,
            batch_size=30,
            train=False,
            sort=False,
            sort_within_batch=False,
            shuffle=False,
        )

        # Translation builder
        translation_builder = TranslationBuilder(
            dataset,
            self._fields,
            n_best,  # NBest
            self._args["replace_unk"],  # replace_unk,
            False,  # batch have gold targets
            self._args["phrase_table"],  # Phrase table
        )

        def _use_gpu(**kwargs):
            return kwargs.get("gpu", -1) >= 0

        results = []
        for i, batch in enumerate(data_iter):
            set_random_seed(
                self._args.get("seed", 829), _use_gpu(**self._args)
            )
            batch_data = self._translator.translate_batch(
                batch, dataset.src_vocabs, True
            )
            translations = translation_builder.from_batch(batch_data)
            for j, translation in enumerate(translations):
                pred = zip(
                    translation.pred_sents,
                    translation.attns,
                    translation.pred_scores
                )
                predictions = []
                for sent, attn, score in pred:
                    sent = (" ").join(sent)
                    attn = attn.tolist()
                    score = score.item()
                    predictions.append(
                        Pred_sent(
                            srcs[i * 30 + j], sent, attn, score
                        )
                    )
                results.append(Translation(predictions))
        return results

    def online_train(self, tus: List[Tuple[str, str]], num_steps: int = 1):
        if self._trainer is not None:
            for _ in range(num_steps):
                # Set model mode to train
                self._opennmt_model.train(True)
                self._opennmt_model.train(True)

                src_reader = inputters.TextDataReader()
                tgt_reader = inputters.TextDataReader()

                # srcs = [tu[0] for tu in tus]
                # tgts = [tu[1] for tu in tus]
                srcs, tgts = map(list, zip(*tus))

                dataset = inputters.Dataset(
                    self._fields,
                    readers=[src_reader, tgt_reader],
                    data=[("src", srcs), ("tgt", tgts)],
                    dirs=[None, None],
                    sort_key=None,
                    filter_pred=None,
                )

                batches = inputters.OrderedIterator(
                    dataset, batch_size=1, device=self._translator._dev
                )

                self._train_steps += 1

                self._trainer.train(
                    batches,
                    self._train_steps,
                    save_checkpoint_steps=None,
                    valid_iter=None,
                    valid_steps=None
                )
        else:
            raise Exception("Online learning not activated for this model.")
