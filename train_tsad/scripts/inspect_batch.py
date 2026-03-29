from __future__ import annotations

from _bootstrap import bootstrap_src_path


bootstrap_src_path()

from train_tsad.cli.inspect_data import main


if __name__ == "__main__":
    main()
