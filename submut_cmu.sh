CMU_SLICE=$1
bsub -n 4 -W 24:00 -R 'rusage[mem=32768]' -J "cmu_"$CMU_SLICE -oo "outputs/output_cmu_"$CMU_SLICE".txt" python run.py --dataset=cmu --input_config=input_configs/default_cmu_subset_$CMU_SLICE.gin --cmu_slice=$CMU_SLICE
