bsub -I -n 1 -R "rusage[mem=32000,ngpus_excl_p=1]" python run_featurePnP.py --dataset /cluster/scratch/aunagar/S2DHM/robotcar/ --result /cluster/scratch/aunagar/S2DHM/result/s2dhm_night-rain_9_14_17_21_24/