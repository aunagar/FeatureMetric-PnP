bsub -I -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" python run.py --dataset robotcar --mode sparse_to_dense
