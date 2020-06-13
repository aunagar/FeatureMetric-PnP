CORES=2
TIME=48:00
MEMORY=16000
CMU_SLICE=21
start=2130
end=2529
bsub -n $CORES -W $TIME -R 'rusage[mem='$MEMORY']' -J "cmu_"$CMU_SLICE"_"$start"_"$end -oo "output_cmu_"$CMU_SLICE"_"$start"_"$end".txt" \
        python run.py \
        --dataset=cmu \
        --input_config=input_configs/default_cmu_subset_$CMU_SLICE.gin \
        --cmu_slice=$CMU_SLICE \
        --cache_results \
        --start=$start \
        --end=$end

