bsub -n 1 -I -R "rusage[mem=32000,ngpus_excl_p=1]" python run_optimizer.py \
    --dataset /nfs/nas12.ethz.ch/fs1201/infk_ivc_students/252-0579-00L/ntselepidis/S2DHM_datasets/RobotCar-Seasons/ \
    --result $SCRATCH/S2DHM_results/
