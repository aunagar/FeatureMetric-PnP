# Feature-metric pose refinement on S2D-Hypercolumn matching

# Run on CMU

Change file s2dhm/configs/datasets/cmu.gin: The line ExtendedCMUDataset.root = '/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/252-0579-00L/psarlin/cmu_extended/' should be adapted to your cmu path!

Run:

python run.py --dataset=cmu --input_config=input_configs/default_cmu_subset_{your id}.gin --cmu_slice={your_id}