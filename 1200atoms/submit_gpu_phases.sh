numj=30
phases=(liquid)
#declare -A LLPT_Tdict=([1700]=$(seq 193 1 195) [1800]=$(seq 184 1 186) [1900]=$(seq 176 1 177) [2000]=$(seq 168 1 171) [2100]=$(seq 161 1 163) [2200]=$(seq 155 1 157) [2300]=$(seq 148 1 150) [2400]=$(seq 142 1 145) [2500]=$(seq 137 1 140))
#declare -A LLPT_Tdict=([1800]="170 190 192 194" [1900]="160 182 184 186" [2000]=$(seq 172 2 184) [2100]=$(seq 170 2 180) [2200]=$(seq 162 10 172) [2300]=$(seq 162 10 172) [2400]=$(seq 150 10 160) [2500]=$(seq 150 10 160))
declare -A LLPT_Tdict=([1700]=$(seq 191.5 1 194.5) [1800]=$(seq 183 0.2 185) [1900]=$(seq 176.2 0.2 177.8) [2000]=$(seq 168 0.2 169.8) [2100]=$(seq 160.2 0.2 164.8) [2200]=$(seq 154.2 0.2 156.8) [2300]=$(seq 148.2 0.2 149.8) [2400]=$(seq 142.5 0.5 145.5))
#declare -A LLPT_Tdict=([1800]=$(seq 183 1 185) [1900]=$(seq 162 4 172) [2000]=$(seq 168 1 169) [2100]=$(seq 161 1 164) [2200]=$(seq 155 1 156) [2300]=$(seq 149 1 149) [2400]=$(seq 143 1 145))
for temp in $(seq 1800 100 2400)
#for p in $(seq 100 10 140)
do
	for ph in ${phases[@]}
	do
		for p in ${LLPT_Tdict[$temp]}
		#for p in 169 170 171
		#for temp in $(seq 2500 100 3000)
		do
			#if p is *.0 then remove .0
			if [[ $p == *".0" ]]; then
				p=${p%.*}
			fi
			case="NPT"
			jobname=$ph
			mkdir -p p$p/$ph/$case/$temp/analysis
			if [ ! -f p$p/1300K_data.txt ]; then
				cp hydrogen_1200atoms_140Gpa_1300K.data p$p/1300K_data.txt
			fi
			cd p$p/$ph/$case/$temp
			ntasks=16
			JOB=$(sbatch \
			-p gpuA40x4 \
			--gpus=1 \
			-N 1 \
			--mem=25g \
		    --ntasks=$ntasks \
			--account=bcqo-delta-gpu \
			-t 02:05:00 \
			--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks \
			-J 1200atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
			--mail-type=NONE \
			../../../../slurm_gpu_phases.sh | tr -cd "[0-9]")
			for j in $(seq 1 1 $numj)
			do
				JOB=$(sbatch \
				-p gpuA40x4 \
				-N 1 \
				--gpus=1 \
				--mem=25g \
				--account=bcqo-delta-gpu \
				--ntasks=$ntasks \
				-t 02:05:00 \
				--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks \
				-J 1200atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
				--mail-type=NONE \
				--dependency=afterany:$JOB \
				../../../../slurm_gpu_phases.sh | tr -cd "[0-9]")
			done
			cd ../../../..
		done
	done
done
