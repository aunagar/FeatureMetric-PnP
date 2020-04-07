bsub -W 48:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output.txt python run.py --dataset robotcar --mode sparse_to_dense
