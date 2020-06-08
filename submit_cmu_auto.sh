#/bin/bash
CORES=2
TIME=00:05
CMU_SLICE=13
NumImages=2
SubSliceSize=2
DivSubSlices=$(( $NumImages / $SubSliceSize))
ModSubSlices=$(( $NumImages % $SubSliceSize))

# First batch
if [ $ModSubSlices -gt 0 ]
then
    start=0
    end=$(($ModSubSlices-1))
    echo $start $end
    bsub -n $CORES -W $TIME -R 'rusage[mem=32768]' -J "cmu_"$CMU_SLICE"_"$start"_"$end -oo "output_cmu_"$CMU_SLICE"_"$start"_"$end".txt" \
        python run.py \
        --dataset=cmu \
        --input_config=input_configs/default_cmu_subset_$CMU_SLICE.gin \
        --cmu_slice=$CMU_SLICE \
        --start=$start \
        --end=$end
fi

# Next batches
for ((i=$ModSubSlices; i<$NumImages; i=$(($i+$SubSliceSize))))
do
    start=$i
    end=$(($i+$SubSliceSize-1))
    echo $start $end
    bsub -n $CORES -W $TIME -R 'rusage[mem=32768]' -J "cmu_"$CMU_SLICE"_"$start"_"$end -oo "output_cmu_"$CMU_SLICE"_"$start"_"$end".txt" \
        python run.py \
        --dataset=cmu \
        --input_config=input_configs/default_cmu_subset_$CMU_SLICE.gin \
        --cmu_slice=$CMU_SLICE \
        --start=$start \
        --end=$end
done
