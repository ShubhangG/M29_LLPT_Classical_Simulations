phases=(liquid)
source /work/nvme/bcqo/shubhanggoswami/mace_modelling/bin/activate
cd /work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms
declare -A LLPT_Tdict=([1700]=$(seq 188 1 198) [1800]=$(seq 180 1 190) [1900]=$(seq 169 1 184) [2000]=$(seq 162 1 174) [2100]=$(seq 150 2 170) [2200]=$(seq 146 2 162) [2300]=$(seq 140 2 156) [2400]=$(seq 136 2 152) [2500]=$(seq 128 2 146))

for temp in $(seq 2100 100 2100)
do
    for ph in ${phases[@]}
    do 
        case=NPT_ASE
        for p in ${LLPT_Tdict[$temp]}
        do
            JOB=$(sbatch \
                -p cpu \
                -N 1 \
                    --ntasks=1 \
                    --account=bcqo-delta-cpu \
                    --export=ALL,temp=$temp,press=$p,phase=$ph,case=$case \
                -t 00:25:00 \
                -J 360_bragg_ASE_${p}_${temp}_${ph} \
                --mem=10g \
                --mail-type=NONE \
                ./run_braggs.sh | tr -cd "[0-9]")
        done
    done
done
