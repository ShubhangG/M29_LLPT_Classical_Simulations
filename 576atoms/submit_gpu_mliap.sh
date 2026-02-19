numj=6
phases=(liquid)
ngpus=1

#declare -A LLPT_Tdict=([1700]=$(seq 191 1 194) [1800]=$(seq 183 0.5 185) [2000]=$(seq 166 1 170) [2100]=$(seq 159 0.5 163) [2300]=$(seq 146 1 150) [2500]=$(seq 134 1 138))
declare -A LLPT_Tdict=([1700]=$(seq 186 2 198) [1800]=$(seq 180 1 188) [1900]=$(seq 169 1 184) [2000]=$(seq 160 2 174) [2100]=$(seq 156 2 168) \
[2200]=$(seq 146 1 162) [2300]=$(seq 140 2 156) [2400]=$(seq 136 2 152) [2500]=$(seq 130 2 142) [3000]=$(seq 96 2 126))

declare -A followup_dict=([1825]="182" [1850]="180")
declare -A forcefiller_dict=([1700]="195 197" [1800]="179 189" [2000]="171 173 175 165" [2100]="157 167 165" [2300]="145 153 155" [2500]="131 133 141 139")

#declare -A leftover_dict=([1800]=$(seq 183.5 1 184.5) [2100]=$(seq 159.5 1 163) [2500]="135")
for temp in 1700 1800 2000 2100 2300 2500
#for temp in 3000
#for temp in 1825 1850
do
	#for p in ${LLPT_Tdict[$temp]}
	#for p in ${followup_dict[$temp]}
	for p in ${forcefiller_dict[$temp]}
	#for p in ${leftover_dict[$temp]}
	#for p in 150
	#for temp in $(seq 2500 100 3000)
	do
		ph=liquid
		#if p is *.0 then remove .0
		if [[ $p == *".0" ]]; then
			p=${p%.*}
		fi
		case="NPT_mliap"
		jobname=$ph
		mkdir -p p$p/$ph/$case/$temp/analysis
		if [ ! -f p$p/2000K_data.txt ]; then
				cp p140/2000K_data.txt p$p/2000K_data.txt
		fi

		cd p$p/$ph/$case/$temp
		# if [ -d "analysis" ]; then
		# 	if ls -v dump.[0-9]*.atom >/dev/null 2>&1; then
		# 		ls -v dump.[0-9]*.atom | xargs cat > analysis/dump.merged.atom
		# 		echo "Dump files merged for P=$p, T=$temp, skipping..."
		# 		cd ../../../..
		# 		continue
		# 	else
		# 		echo "ERROR: No dump files found for P=$p, T=$temp" >&2
		# 		echo "Running MLIAP for P=$p, T=$temp"
		# 	fi
		# fi
		ntasks=16
		JOB=$(sbatch \
		-p gpuA100x4 \
		--gpus=$ngpus \
		-N 1 \
		--mem=10g \
		--ntasks=$ntasks \
		--account=bcqo-delta-gpu \
		-t 01:00:00 \
		--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks,ngpus=$ngpus \
		-J MLIAP_576atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
		--mail-type=NONE \
		../../../../slurm_gpu_mliap_rhel9.sh | tr -cd "[0-9]")
		for j in $(seq 1 1 $numj)
		do
			JOB=$(sbatch \
			-p gpuA100x4 \
			-N 1 \
			--gpus=$ngpus \
			--mem=10g \
			--account=bcqo-delta-gpu \
			--ntasks=$ntasks \
			-t 01:00:00 \
			--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks,ngpus=$ngpus \
			-J MLIAP_576atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
			--mail-type=NONE \
			--dependency=afterany:$JOB \
			../../../../slurm_gpu_mliap_rhel9.sh | tr -cd "[0-9]")
		done
		cd ../../../..
	done
done
