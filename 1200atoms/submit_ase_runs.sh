numj=8
phases=(liquid)
#declare -A LLPT_Tdict=([1700]=$(seq 186 2 200) [1800]=$(seq 180 1 190) [1900]=$(seq 169 1 184) [2000]=$(seq 160 1 180) [2100]=$(seq 156 2 170) [2200]=$(seq 146 1 162) [2300]=$(seq 140 2 160) [2400]=$(seq 136 2 152) [2500]=$(seq 130 2 150))
#declare -A LLPT_Tdict=([1700]=$(seq 192.5 1 193.5) [1800]=$(seq 183 0.2 185) [1900]=$(seq 176.2 0.2 177.8) [2000]=$(seq 167 0.2 169.8) [2100]=$(seq 160.2 0.2 164) [2200]=$(seq 154.2 0.2 156.8) [2300]=$(seq 146.2 0.2 149.8) [2400]=$(seq 142.5 0.5 145.5) [2500]=$(seq 134.5 1 137.5))
declare -A LLPT_Tdict=([1700]=$(seq 192 0.5 194) [1800]=$(seq 183 0.2 185) [2000]=$(seq 167.4 0.2 168.8) [2100]=$(seq 160 0.2 162) [2300]=$(seq 146.2 0.2 149.4) [2500]=$(seq 134 0.5 138))

#for temp in $(seq 1700 100 2500)
for temp in 1700 1800 2000 2100 2300 2500
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
        if [ ! -f p$p/1300K_data.txt ]; then
            cp hydrogen_1200atoms_140Gpa_1300K.data p$p/1300K_data.txt
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
        -t 04:00:00 \
        --export=ALL,temp=$temp,run_type=$run_type,press=$p \
        -J ASE_1200atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
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
            -t 04:00:00 \
            --export=ALL,temp=$temp,run_type=$run_type,press=$p \
            -J ASE_1200atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
            --mail-type=NONE \
            --dependency=afterany:$JOB \
            ../../../../run_ase.sh | tr -cd "[0-9]")
        done
        cd ../../../..
    done 
done 
 