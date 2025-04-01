This is a fork of the [Obsidian](https://github.com/gab8192/Obsidian/) chess engine, that I modified for a [competition](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge) on Kaggle. My writeup is available [HERE](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/571067).

Compilation commands:
```bash
make nopgo # Compile ./Obsidian with an UCI interface, using the nnue net.nn
make nopgo shared=1 # Compile a shared library ./Obsidian.so
make nopgo shared=1 quant=1 # Compile a shared library ./Obsidian.so, using the requantized nnue nn_quant.net

make clean; make -j nopgo shared=1 quant=1; strip Obsidian.so; cp Obsidian.so O.so; xz -9fk O.so; tar cz main.py O.so.xz -f sub.tar.gz; ls -la # Create the submission file
```

