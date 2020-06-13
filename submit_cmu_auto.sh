#/bin/bash
CORES=2
MEMORY=16768 # Per core
TIME=24:00
CMU_SLICE=5
NumImages=2370
SubSliceSize=200
DivSubSlices=$(( $NumImages / $SubSliceSize))
ModSubSlices=$(( $NumImages % $SubSliceSize))

# First batch
if [ $ModSubSlices -gt 0 ]
then
    start=0
    end=$(($ModSubSlices-1))
    echo $start $end
    bsub -n $CORES -W $TIME -R 'rusage[mem='$MEMORY']' -J "cmu_"$CMU_SLICE"_"$start"_"$end -oo "output_cmu_"$CMU_SLICE"_"$start"_"$end".txt" \
        python run.py \
        --dataset=cmu \
        --input_config=input_configs/default_cmu_subset_$CMU_SLICE.gin \
        --cmu_slice=$CMU_SLICE \
	--cache_results \
        --start=$start \
        --end=$end
fi

# Next batches
for ((i=$ModSubSlices; i<$NumImages; i=$(($i+$SubSliceSize))))
do
    start=$i
    end=$(($i+$SubSliceSize-1))
    echo $start $end
    bsub -n $CORES -W $TIME -R 'rusage[mem='$MEMORY']' -J "cmu_"$CMU_SLICE"_"$start"_"$end -oo "output_cmu_"$CMU_SLICE"_"$start"_"$end".txt" \
        python run.py \
        --dataset=cmu \
        --input_config=input_configs/default_cmu_subset_$CMU_SLICE.gin \
        --cmu_slice=$CMU_SLICE \
	--cache_results \
        --start=$start \
        --end=$end
done
