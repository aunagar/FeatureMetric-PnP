bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_original_hcs_no_outliers.txt python run.py --dataset robotcar --mode sparse_to_dense
