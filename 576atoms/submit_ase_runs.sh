numj=8
phases=(liquid)
#declare -A LLPT_Tdict=([1700]=$(seq 186 2 200) [1800]=$(seq 180 1 190) [1900]=$(seq 169 1 184) [2000]=$(seq 160 1 180) [2100]=$(seq 156 2 170) [2200]=$(seq 146 1 162) [2300]=$(seq 140 2 160) [2400]=$(seq 136 2 152) [2500]=$(seq 130 2 150))
#declare -A LLPT_Tdict=([1700]=$(seq 188 0.5 196) [1800]=$(seq 182 0.2 185) [1900]=$(seq 174.5 0.5 182.5) [2000]=$(seq 165.5 0.5 171.5) [2100]=$(seq 158 0.5 164) [2200]=$(seq 151.5 0.5 161.5) [2300]=$(seq 143 0.5 152) [2400]=$(seq 150 10 160) [2500]=$(seq 136.5 0.5 141))
declare -A LLPT_Tdict=([1700]=$(seq 191 1 194) [1800]=$(seq 183 0.5 185) [2000]=$(seq 166 1 170) [2100]=$(seq 159 0.5 163) [2300]=$(seq 146 1 150) [2500]=$(seq 134 1 138))

#for temp in $(seq 1700 100 2100)
for temp in 1700 1800 2000 2100 2300 2500
#for temp in 1800 2000
do
    for p in ${LLPT_Tdict[$temp]}
    #for p in 170
    do
    	if [[ $p == *".0" ]]; then
			p=${p%.*}
		fi
        ph="liquid"
        case="NPT_ASE"
        run_type="NPT"
	    jobname=$ph
        mkdir -p p$p/$ph/$case/$temp/analysis
        if [ ! -f p$p/2000K_data.txt ]; then
            cp p140/2000K_data.txt p$p/2000K_data.txt
        fi
        cd p$p/$ph/$case/$temp        
        ntasks=16
        JOB=$(sbatch \
        -p gpuA100x4 \
        --gpus=1 \
        -N 1 \
        --mem=10g \
        --ntasks=$ntasks \
        --account=bcqo-delta-gpu \
        -t 02:00:00 \
        --export=ALL,temp=$temp,run_type=$run_type,press=$p \
        -J ASE_576atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
        --mail-type=NONE \
        ../../../../run_ase.sh | tr -cd "[0-9]")
        run_num=$((run_num+1))
        for j in $(seq 1 1 $numj)
        do
            JOB=$(sbatch \
            -p gpuA100x4 \
            -N 1 \
            --gpus=1 \
            --mem=10g \
            --account=bcqo-delta-gpu \
            --ntasks=$ntasks \
            -t 02:00:00 \
            --export=ALL,temp=$temp,run_type=$run_type,press=$p \
            -J ASE_576atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
            --mail-type=NONE \
            --dependency=afterany:$JOB \
            ../../../../run_ase.sh | tr -cd "[0-9]")
        done
        cd ../../../..
    done 
done 
 