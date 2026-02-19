numj=0
phases=(liquid)
#declare -A LLPT_Tdict=([1700]=$(seq 190 1 200) [1800]=$(seq 180 1 190) [1900]=$(seq 160 2 190) [2000]=$(seq 160 2 180) [2100]=$(seq 160 1 180) [2200]=$(seq 150 2 170) [2300]=$(seq 146 2 160) [2400]=$(seq 142 2 160) [2500]=$(seq 130 2 160) [3000]=$(seq 100 5 140))
declare -A LLPT_Tdict=([1900]=$(seq 173 2 180) [2000]=$(seq 167 2 172) [2100]=$(seq 150 2 159) [2200]="145 147 155 157" [2300]=$(seq 140 2 144) [2400]=$(seq 136 2 140))
#declare -A LLPT_Tdict=([1700]=$(seq 191.5 1 194.5) [1800]=$(seq 183 0.2 185) [1900]="160 182 184 186" [2000]=$(seq 172 2 184) [2100]=$(seq 170 2 180) [2200]=$(seq 162 10 172) [2300]=$(seq 162 10 172) [2400]=$(seq 150 10 160) [2500]=$(seq 150 10 160))
for temp in $(seq 2100 100 2100)
#for temp in 3000
#for p in $(seq 100 10 140)
do
	for ph in ${phases[@]}
	do
		#for p in ${LLPT_Tdict[$temp]}
		for p in 170
		#for temp in $(seq 2500 100 3000)
		do
			case="NPT_new"
			jobname=$ph
			mkdir -p p$p/$ph/$case/$temp/analysis
			if [ ! -f p$p/data_liquid.txt ]; then
				cp hydrogen_360_atom_2000K_liquiddata.txt p$p/data_liquid.txt
			fi
			cd p$p/$ph/$case/$temp
			ntasks=16
			JOB=$(sbatch \
			-p gpuA40x4 \
			--gpus=1 \
			-N 1 \
			--mem=64g \
		    --ntasks=$ntasks \
			--account=bcqo-delta-gpu \
			-t 02:00:00 \
			--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks \
			-J 360atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
			--mail-type=NONE \
			../../../../slurm_gpu_phases.sh | tr -cd "[0-9]")
			for j in $(seq 1 1 $numj)
			do
				JOB=$(sbatch \
				-p gpuA40x4 \
				-N 1 \
				--gpus=1 \
				--mem=10g \
				--account=bcqo-delta-gpu \
				--ntasks=$ntasks \
				-t 02:00:00 \
				--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks \
				-J 360atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
				--mail-type=NONE \
				--dependency=afterany:$JOB \
				../../../../slurm_gpu_phases.sh | tr -cd "[0-9]")
			done
			cd ../../../..
		done
	done
done
