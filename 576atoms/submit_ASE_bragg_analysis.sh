phases=(liquid)
source /work/nvme/bcqo/shubhanggoswami/mace_modelling/bin/activate
cd /work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/576atoms
#declare -A LLPT_Tdict=([1700]=$(seq 186 2 200) [1800]=$(seq 180 1 190) [1900]=$(seq 169 1 184) [2000]=$(seq 160 1 180) [2100]=$(seq 156 2 170) [2200]=$(seq 146 1 162) [2300]=$(seq 140 2 160) [2400]=$(seq 136 2 152) [2500]=$(seq 130 2 150))
declare -A LLPT_Tdict=([1700]=$(seq 188 0.5 196) [1800]=$(seq 182 0.2 185) [1900]=$(seq 174.5 0.5 182.5) [2000]=$(seq 165.5 0.5 171.5) [2100]=$(seq 158 0.5 164) [2200]=$(seq 151.5 0.5 161.5) [2300]=$(seq 143 1 152) [2400]=$(seq 150 10 160) [2500]=$(seq 136.5 0.5 141))

for temp in 2300
#for temp in 1700 1800 2000 2100 2300 2500
do
    for ph in ${phases[@]}
    do 
        case=NPT_ASE
        for p in ${LLPT_Tdict[$temp]}
        #for p in $(seq 100 5 140)
        #for p in 150 170
        do
            if [[ $p == *".0" ]]; then
			    p=${p%.*}
            else 
                continue
		    fi
            JOB=$(sbatch \
                -p cpu \
                -N 1 \
                    --ntasks=1 \
                    --account=bcqo-delta-cpu \
                    --export=ALL,temp=$temp,press=$p,phase=$ph,case=$case \
                -t 00:25:00 \
                -J 576_bragg_ASE_${p}_${temp}_${ph} \
                --mem=10g \
                --mail-type=NONE \
                ./run_braggs.sh | tr -cd "[0-9]")
        done
    done
done
