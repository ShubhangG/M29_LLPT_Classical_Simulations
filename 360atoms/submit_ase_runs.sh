numj=4
phases=(liquid)
#declare -A LLPT_Tdict=([1700]=$(seq 188 1 198) [1800]=$(seq 180 1 190) [1900]=$(seq 169 1 184) [2000]=$(seq 162 1 174) [2100]=$(seq 150 2 170) [2200]=$(seq 146 2 162) [2300]=$(seq 140 2 156) [2400]=$(seq 136 2 152) [2500]=$(seq 128 2 146))
declare -A LLPT_Tdict=([1700]=$(seq 192 1 195) [1800]=$(seq 183 1 186) [2000]=$(seq 166 1 170) [2100]=$(seq 150 2 170) [2300]=$(seq 146 1 150) [2500]=$(seq 134 1 139))

#for temp in $(seq 1700 100 2500)
for temp in 1700 1800 2000 2100 2300 2500
do
    for p in ${LLPT_Tdict[$temp]}
    #for p in 170
    do
        ph="liquid"
        case="NPT_ASE"
        run_type="NPT"
	    jobname=$ph
        mkdir -p p$p/$ph/$case/$temp/analysis
        if [ ! -f p$p/data_liquid.txt ]; then
            cp hydrogen_360_atom_2000K_liquiddata.txt p$p/data_liquid.txt
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
        -J ASE_360atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
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
            -J ASE_360atom_M29_${p}_${temp}_$(echo $case | head -c 3) \
            --mail-type=NONE \
            --dependency=afterany:$JOB \
            ../../../../run_ase.sh | tr -cd "[0-9]")
        done
        cd ../../../..
    done 
done 
 