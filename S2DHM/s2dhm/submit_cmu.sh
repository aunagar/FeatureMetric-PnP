CMU_SLICE=$1
bsub -W 60:00 -n 1 -R "rusage[mem=22768,ngpus_excl_p=1]" -oo output_cmu_slice_$CMU_SLICE.txt python run.py --dataset cmu --mode sparse_to_dense --cmu_slice $CMU_SLICE
