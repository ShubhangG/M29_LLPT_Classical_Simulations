numj=9
phases=(liquid)
ngpus=2
#declare -A LLPT_Tdict=([1700]=$(seq 193 1 195) [1800]=$(seq 184 1 186) [1900]=$(seq 176 1 177) [2000]=$(seq 168 1 171) [2100]=$(seq 161 1 163) [2200]=$(seq 155 1 157) [2300]=$(seq 148 1 150) [2400]=$(seq 142 1 145) [2500]=$(seq 137 1 140))
#declare -A LLPT_Tdict=([1800]="170 190 192 194" [1900]="160 182 184 186" [2000]=$(seq 172 2 184) [2100]=$(seq 170 2 180) [2200]=$(seq 162 10 172) [2300]=$(seq 162 10 172) [2400]=$(seq 150 10 160) [2500]=$(seq 150 10 160))
#declare -A LLPT_Tdict=([1700]=$(seq 191.5 1 194.5) [1800]=$(seq 183 0.2 185) [1900]=$(seq 176.2 0.2 177.8) [2000]=$(seq 168 0.2 169.8) [2100]=$(seq 160.2 0.2 164.8) [2200]=$(seq 154.2 0.2 156.8) [2300]=$(seq 148.2 0.2 149.8) [2400]=$(seq 142.5 0.5 145.5) [2500]=$(seq 136.5 1 139.5))
#declare -A LLPT_Tdict=([1800]=$(seq 183 1 185) [1900]=$(seq 162 4 172) [2000]=$(seq 168 1 169) [2100]=$(seq 161 1 164) [2200]=$(seq 155 1 156) [2300]=$(seq 149 1 149) [2400]=$(seq 143 1 145))
#declare -A LLPT_Tdict=([1700]=$(seq 192 0.5 194) [1800]=$(seq 183 0.2 185) [2000]=$(seq 167.4 0.2 168.8) [2100]=$(seq 160 0.2 162) [2300]=$(seq 146.2 0.2 149.4) [2500]=$(seq 134 0.5 138))
# declare -A leftover_Tdict=([1700]="192.5 193 193.5" [1800]="$(seq 183.2 0.2 183.8) $(seq 184.2 0.2 184.8)" [2000]="$(seq 167.4 0.2 167.8) $(seq 168.2 0.2 168.8)" [2100]="$(seq 160.2 0.2 160.8) $(seq 161.2 0.2 161.8)" \
# [2300]="$(seq 146.2 0.2 146.8) $(seq 147.2 0.2 147.8) $(seq 148.2 0.2 148.8) $(seq 149.2 0.2 149.4)" [2500]="135 135.5 136.5 137 137.5" [3000]=$(seq 96 2 126))
# declare -A LLPT_Tdict=([1700]="186 188 190 196 198" [1800]="180 182 186 188" [2000]=$(seq 162 2 166 ; seq 170 2 174) [2100]="156 158 163 164 166" \
#  [2300]="142 144 150 152" [2500]="130 132 140 142")


#for temp in $(seq 1900 100 1900)
#for p in $(seq 100 10 140)
#for temp in 1700 1800 2000 2100 2300 2500
# for temp in 1700
# do
	for ph in ${phases[@]}
	do
		temp=1925
		p=174
		#for p in ${LLPT_Tdict[$temp]}
		#for p in ${leftover_Tdict[$temp]}
		#for p in 192.25 192.75 193.25 193.75
		#for p in 172 
		#for temp in $(seq 2500 100 3000)
		#do
			#if p is *.0 then remove .0
			if [[ $p == *".0" ]]; then
				p=${p%.*}
			fi
			case="NPT_mliap"
			jobname=$ph
			mkdir -p p$p/$ph/$case/$temp/analysis
			if [ ! -f p$p/1300K_data.txt ]; then
				cp hydrogen_1200atoms_140Gpa_1300K.data p$p/1300K_data.txt
			fi
			cd p$p/$ph/$case/$temp
			ntasks=16
			JOB=$(sbatch \
			-p gpuA40x4 \
			--gpus=$ngpus \
			-N 1 \
			--mem=10g \
		    --ntasks=$ntasks \
			--account=bcqo-delta-gpu \
			-t 02:30:00 \
			--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks,ngpus=$ngpus \
			-J MLIAP_1200atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
			--mail-type=NONE \
			../../../../slurm_gpu_mliap.sh | tr -cd "[0-9]")
			for j in $(seq 1 1 $numj)
			do
				JOB=$(sbatch \
				-p gpuA40x4 \
				-N 1 \
				--gpus=$ngpus \
				--mem=10g \
				--account=bcqo-delta-gpu \
				--ntasks=$ntasks \
				-t 02:30:00 \
				--export=ALL,temp=$temp,case=$case,press=$p,phase=$ph,ntasks=$ntasks,ngpus=$ngpus \
				-J MLIAP_1200atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
				--mail-type=NONE \
				--dependency=afterany:$JOB \
				../../../../slurm_gpu_mliap.sh | tr -cd "[0-9]")
			done
			cd ../../../..
		#done
	done
#done
