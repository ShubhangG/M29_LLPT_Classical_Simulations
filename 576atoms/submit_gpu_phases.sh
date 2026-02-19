numj=25
phases=(liquid)
#declare -A LLPT_Tdict=([1700]=$(seq 193 1 196) [1800]=$(seq 184 1 186) [1900]=$(seq 175 1 177) [2000]=$(seq 168 1 171) [2100]=$(seq 160 1 166) [2200]=$(seq 155 1 157) [2300]=$(seq 148 1 151) [2400]=$(seq 141 1 145) [2500]=$(seq 136 1 140))
declare -A LLPT_Tdict=([1700]=$(seq 185 0.2 196) [1800]=$(seq 183 0.2 185) [1900]=$(seq 174.5 0.5 182.5) [2000]=$(seq 165.5 0.5 171.5) [2100]=$(seq 161.5 0.5 171) [2200]=$(seq 151.5 0.5 161.5) [2300]=$(seq 162 10 172) [2400]=$(seq 150 10 160) [2500]=$(seq 150 10 160))
for temp in $(seq 1700 100 2200)
#for p in $(seq 100 10 140)
do
	for ph in ${phases[@]}
	do
		for p in ${LLPT_Tdict[$temp]}
		#for p in 169 170 171
		#for temp in $(seq 2500 100 3000)
		do
			if [[ $p == *".0" ]]; then
				p=${p%.*}
			fi
			case="NPT"
			jobname=$ph
			mkdir -p p$p/$ph/$case/$temp/analysis
			if [ ! -f p$p/2000K_data.txt ]; then
				cp p140/2000K_data.txt p$p/2000K_data.txt
			fi
			cd p$p/$ph/$case/$temp
			ntasks=16
			JOB=$(sbatch \
			-p gpuA40x4 \
			--gpus=1 \
			-N 1 \
			--mem=12g \
		    --ntasks=$ntasks \
			--account=bcqo-delta-gpu \
			-t 02:25:00 \
			--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks \
			-J 576atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
			--mail-type=NONE \
			../../../../slurm_gpu_phases.sh | tr -cd "[0-9]")
			for j in $(seq 1 1 $numj)
			do
				JOB=$(sbatch \
				-p gpuA40x4 \
				-N 1 \
				--gpus=1 \
				--mem=12g \
				--account=bcqo-delta-gpu \
				--ntasks=$ntasks \
				-t 02:25:00 \
				--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks \
				-J 576atom_M29${p}_${temp}_$(echo $case | head -c 3) \
				--mail-type=NONE \
				--dependency=afterany:$JOB \
				../../../../slurm_gpu_phases.sh | tr -cd "[0-9]")
			done
			cd ../../../..
		done
	done
done
