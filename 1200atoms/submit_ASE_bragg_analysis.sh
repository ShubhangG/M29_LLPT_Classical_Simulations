phases=(liquid)
source /work/nvme/bcqo/shubhanggoswami/mace_modelling/bin/activate
cd /work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/1200atoms
#declare -A LLPT_Tdict=([1700]=$(seq 186 2 200) [1800]=$(seq 180 1 190) [1900]=$(seq 169 1 184) [2000]=$(seq 160 1 180) [2100]=$(seq 156 2 170) [2200]=$(seq 146 1 162) [2300]=$(seq 140 2 160) [2400]=$(seq 136 2 152) [2500]=$(seq 130 2 150))
declare -A LLPT_Tdict=([1700]=$(seq 192.5 1 193.5) [1800]=$(seq 183 0.2 185) [1900]=$(seq 176.2 0.2 177.8) [2000]=$(seq 167 0.2 169.8) [2100]=$(seq 160.2 0.2 164) [2200]=$(seq 154.2 0.2 156.8) [2300]=$(seq 146.2 0.2 149.8) [2400]=$(seq 142.5 0.5 145.5) [2500]=$(seq 134.5 1 137.5))

for temp in 1700 1800 2000 2100 2300 2500
#for temp in 1700
do
    for ph in ${phases[@]}
    do 
        case=NPT_ASE
        for p in ${LLPT_Tdict[$temp]}
        do
            if [[ $p == *".0" ]]; then
			    p=${p%.*}
		    fi
            JOB=$(sbatch \
                -p cpu \
                -N 1 \
                    --ntasks=1 \
                    --account=bcqo-delta-cpu \
                    --export=ALL,temp=$temp,press=$p,phase=$ph,case=$case \
                -t 01:20:00 \
                -J 1200_bragg_ASE_${p}_${temp}_${ph} \
                --mem=10g \
                --mail-type=NONE \
                ./run_braggs.sh | tr -cd "[0-9]")
        done
    done
done
