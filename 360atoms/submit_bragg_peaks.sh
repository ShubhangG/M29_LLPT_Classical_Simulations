phases=(liquid)
source /work/nvme/bcqo/shubhanggoswami/mace_modelling/bin/activate
cd /work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/360atoms
#declare -A LLPT_Tdict=([1700]=$(seq 190 1 200) [1800]=$(seq 180 1 190) [1900]=$(seq 160 2 190) [2000]=$(seq 160 2 180) [2100]=$(seq 160 1 180) [2200]=$(seq 150 2 170) [2300]=$(seq 146 2 160) [2400]=$(seq 142 2 160) [2500]=$(seq 130 2 160))
#declare -A LLPT_Tdict=([1900]=$(seq 173 2 180) [2000]=$(seq 167 2 172) [2100]=$(seq 150 2 159) [2200]="145 147 155 157" [2300]=$(seq 140 2 144) [2400]=$(seq 136 2 140))
#declare -A LLPT_Tdict=([1700]=$(seq 192 1 195) [1800]=$(seq 183 1 186) [2000]=$(seq 166 1 170) [2100]=$(seq 150 2 170) [2300]=$(seq 146 1 150) [2500]=$(seq 134 1 139))
declare -A LLPT_Tdict=([1700]=$(seq 188 1 198) [1800]=$(seq 180 1 190) [1900]=$(seq 169 1 184) [2000]=$(seq 162 1 165; seq 171 1 174) [2100]=$(seq 157 2 167) \
[2200]=$(seq 146 2 162) [2300]=$(seq 140 2 144; seq 152 2 156) [2400]=$(seq 136 2 152) [2500]=$(seq 128 2 134 ; seq 140 2 146) [3000]=$(seq 96 2 126))
declare -A leftover_dict=([2300]=$(seq 147 2 149) [2500]=$(seq 135 2 139))
# for temp in $(seq 1900 100 2400)
#for temp in 1700 1800 2000 2100 2300 2500 3000
for temp in 1750
do
    for ph in ${phases[@]}
    do 
        case=NPT_mliap_new
        #for p in ${LLPT_Tdict[$temp]}
        #for p in ${leftover_dict[$temp]}
        #for p in $(seq 100 5 140)
        for p in 190
        do
            #temp=2100
            python ./lammps_out_reader_phases.py -p $p -t $temp -f $ph -c $case -n 360
            cd p$p/$ph/$case/$temp/
            if [ -d "analysis" ]; then
                if ls -v dump.[0-9]*.atom >/dev/null 2>&1; then
                    ls -v dump.[0-9]*.atom | xargs cat > analysis/dump.merged.atom
                else
                    echo "ERROR: No dump files found for P=$p, T=$temp" >&2
                    cd ../../../../
                    continue
                fi
            fi
            cd ../../../../
            JOB=$(sbatch \
                -p cpu \
                -N 1 \
                    --ntasks=1 \
                    --account=bcqo-delta-cpu \
                    --export=ALL,temp=$temp,press=$p,phase=$ph,case=$case \
                -t 00:45:00 \
                -J 360_bragg_mliap_${p}_${temp}_${ph} \
                --mem=10g \
                --mail-type=NONE \
                ./run_braggs.sh | tr -cd "[0-9]")
        done
    done
done

