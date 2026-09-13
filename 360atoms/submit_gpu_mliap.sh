numj=10
phases=(liquid)
ngpus=1
#declare -A LLPT_Tdict=([1700]=$(seq 193 1 195) [1800]=$(seq 184 1 186) [1900]=$(seq 176 1 177) [2000]=$(seq 168 1 171) [2100]=$(seq 161 1 163) [2200]=$(seq 155 1 157) [2300]=$(seq 148 1 150) [2400]=$(seq 142 1 145) [2500]=$(seq 137 1 140))
#declare -A LLPT_Tdict=([1800]="170 190 192 194" [1900]="160 182 184 186" [2000]=$(seq 172 2 184) [2100]=$(seq 170 2 180) [2200]=$(seq 162 10 172) [2300]=$(seq 162 10 172) [2400]=$(seq 150 10 160) [2500]=$(seq 150 10 160))
#declare -A LLPT_Tdict=([1700]=$(seq 191.5 1 194.5) [1800]=$(seq 183 0.2 185) [1900]=$(seq 176.2 0.2 177.8) [2000]=$(seq 168 0.2 169.8) [2100]=$(seq 160.2 0.2 164.8) [2200]=$(seq 154.2 0.2 156.8) [2300]=$(seq 148.2 0.2 149.8) [2400]=$(seq 142.5 0.5 145.5) [2500]=$(seq 136.5 1 139.5))
#declare -A LLPT_Tdict=([1800]=$(seq 183 1 185) [1900]=$(seq 162 4 172) [2000]=$(seq 168 1 169) [2100]=$(seq 161 1 164) [2200]=$(seq 155 1 156) [2300]=$(seq 149 1 149) [2400]=$(seq 143 1 145))
#declare -A LLPT_Tdict=([1700]=$(seq 192 1 195) [1800]=$(seq 183 1 186) [2000]=$(seq 166 1 170) [2100]=$(seq 150 2 170) [2300]=$(seq 146 1 150) [2500]=$(seq 134 1 139))
#declare -A LLPT_Tdict=([1700]=$(seq 188 1 198) [1800]=$(seq 180 1 190) [1900]=$(seq 169 1 184) [2000]=$(seq 162 1 165; seq 171 1 174) [2100]=$(seq 157 2 167) \
#[2200]=$(seq 146 2 162) [2300]=$(seq 140 2 144; seq 152 2 156) [2400]=$(seq 136 2 152) [2500]=$(seq 128 2 134 ; seq 140 2 146) [3000]=$(seq 96 2 126))
#declare -A leftover_dict=([2300]=$(seq 147 2 149) [2500]=$(seq 135 2 139))

#for p in $(seq 100 10 140)
#for temp in $(seq 2300 200 2500)
#for temp in 1700 1800 2000 2100 2300 2500
for temp in 1815
#for temp in 1755 1775 1795
do
	for ph in ${phases[@]}
	do
		#for p in ${LLPT_Tdict[$temp]}
		#for p in ${leftover_dict[$temp]}
		#for p in 190
		#for temp in $(seq 2500 100 3000)
		for p in 184
		do
			#if p is *.0 then remove .0
			if [[ $p == *".0" ]]; then
				p=${p%.*}
			fi
			case="NPT_mliap"
			jobname=$ph
			mkdir -p p$p/$ph/$case/$temp/analysis
			if [ ! -f p$p/data_liquid.txt ]; then
				cp hydrogen_360_atom_2000K_liquiddata_cubic.txt p$p/data_liquid.txt
			fi
			cd p$p/$ph/$case/$temp
			ntasks=16
			JOB=$(sbatch \
			-p gpuA100x4 \
			--gpus=$ngpus \
			-N 1 \
			--mem=10g \
		    --ntasks=$ntasks \
			--account=bcqo-delta-gpu \
			-t 00:45:00 \
			--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks,ngpus=$ngpus \
			-J MLIAP_360atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
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
				-t 01:15:00 \
				--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks,ngpus=$ngpus \
				-J MLIAP_360atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
				--mail-type=NONE \
				--dependency=afterany:$JOB \
				../../../../slurm_gpu_mliap_rhel9.sh | tr -cd "[0-9]")
			done
			cd ../../../..
		done
	done
done
