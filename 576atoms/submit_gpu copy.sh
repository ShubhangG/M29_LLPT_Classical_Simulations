numj=5
#for p in $(seq 100 25 100)
for p in 150
do
	for case in NPT
	do
		for temp in $(seq 1600 50 1700)
		do
			mkdir -p p$p/$case/$temp/analysis
			cd p$p/$case/$temp
			ntasks=16
			JOB=$(sbatch \
			-p gpuA40x4 \
			--gpus=1 \
			-N 1 \
			--mem=10g \
		    --ntasks=$ntasks \
			--account=bcqo-delta-gpu \
			-t 03:00:00 \
			--export=ALL,temp=$temp,case=$case,press=$p,ntasks=$ntasks \
			-J 96atom_${p}_${temp}_$(echo $case | head -c 3) \
			--mail-type=NONE \
			../../../slurm_gpu.sh | tr -cd "[0-9]")
			for j in $(seq 1 1 $numj)
			do
				JOB=$(sbatch \
				-p gpuA40x4 \
				-N 1 \
				--gpus=1 \
				--mem=10g \
				--account=bcqo-delta-gpu \
				--ntasks=$ntasks \
				-t 03:00:00 \
				--export=ALL,temp=$temp,case=$case,press=$p,ntasks=$ntasks \
				-J 96atom_${p}_${temp}_$(echo $case | head -c 3) \
				--mail-type=NONE \
				--dependency=afterany:$JOB \
				../../../slurm_gpu.sh | tr -cd "[0-9]")
			done
			cd ../../..
		done
	done
done
