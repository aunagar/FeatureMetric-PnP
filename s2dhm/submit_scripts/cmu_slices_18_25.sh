bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_cmu_slice18.txt python run.py --dataset cmu --mode sparse_to_dense --cmu_slice 18
bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_cmu_slice19.txt python run.py --dataset cmu --mode sparse_to_dense --cmu_slice 19
bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_cmu_slice20.txt python run.py --dataset cmu --mode sparse_to_dense --cmu_slice 20
bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_cmu_slice21.txt python run.py --dataset cmu --mode sparse_to_dense --cmu_slice 21
bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_cmu_slice22.txt python run.py --dataset cmu --mode sparse_to_dense --cmu_slice 22
bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_cmu_slice23.txt python run.py --dataset cmu --mode sparse_to_dense --cmu_slice 23
bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_cmu_slice24.txt python run.py --dataset cmu --mode sparse_to_dense --cmu_slice 24
bsub -W 60:00 -n 1 -R "rusage[mem=32768,ngpus_excl_p=1]" -oo output_cmu_slice25.txt python run.py --dataset cmu --mode sparse_to_dense --cmu_slice 25