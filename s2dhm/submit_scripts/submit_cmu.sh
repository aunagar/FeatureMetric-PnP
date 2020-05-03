bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_cmu_slice2.txt python run.py --dataset cmu --mode sparse_to_dense
