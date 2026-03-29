from __future__ import annotations

from _bootstrap import bootstrap_src_path


bootstrap_src_path()

from train_tsad.cli.train import main


if __name__ == "__main__":
    main()
