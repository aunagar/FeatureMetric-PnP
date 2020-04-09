bsub -W 12:00 -n 1 -R "rusage[mem=12768,ngpus_excl_p=1]" -oo output_night_rain.txt python run.py --dataset robotcar --mode sparse_to_dense
