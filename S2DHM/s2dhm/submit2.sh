bsub -W 12:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_night_rain_914_17_21_24.txt python run.py --dataset robotcar --mode sparse_to_dense
