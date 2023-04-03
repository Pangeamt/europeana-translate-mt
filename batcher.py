from onmt.bin.preprocess import preprocess, _get_parser


def batch(stage_dir: str, *args, **kwargs):
    parser = _get_parser()
    opt = parser.parse_args(list(args))
    preprocess(opt)
