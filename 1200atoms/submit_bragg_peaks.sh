phases=(liquid)
source /work/nvme/bcqo/shubhanggoswami/mace_modelling/bin/activate
cd /work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/1200atoms
# temps_dic2=([25]="$(seq 600 20 800)" [50]="$(seq 800 20 880)" [60]="$(seq 1003 20 1103)" [70]="$(seq 1072 20 1172)" [80]="$(seq 1150 10 1250)" [90]="$(seq 1222 10 1322)" [100]="$(seq 1300 10 1400)" [110]="$(seq 1360 10 1460)" [120]="$(seq 1425 10 1525)" [130]="$(seq 1486 10 1586)" [140]="$(seq 1545 10 1645)" [150]="$(seq 1600 10 1700)" \
# 			[160]="$(seq 1652 10 1752)" [170]="$(seq 1701 10 1801)" [180]="$(seq 1748 10 1848)" [200]="$(seq 1500 50 1800)" [210]="$(seq 1400 50 1700)" [220]="$(seq 1300 50 1600)" [230]="$(seq 1200 50 1500)" [240]="$(seq 1100 50 1400)" [250]="$(seq 1000 50 1300)")
#declare -A LLPT_Tdict=([1700]=$(seq 191 2 199) [1800]=$(seq 181 2 189) [1900]=$(seq 171 2 179) [2000]=$(seq 161 2 179) [2100]=$(seq 161 2 169) [2200]=$(seq 151 2 159) [2300]=$(seq 141 2 149) [2400]=$(seq 141 2 149))
#declare -A LLPT_Tdict=([1700]=$(seq 191 1 195) [1800]=$(seq 181 1 189) [1900]=$(seq 171 1 179) [2000]=$(seq 164 2 168) [2100]=$(seq 156 1 161) [2200]=$(seq 151 1 158) [2300]=$(seq 148 1 158) [2400]=$(seq 138 1 148) [2500]=$(seq 131 2 149))
#declare -A LLPT_Tdict=([1700]=$(seq 190 1 196) [1800]=$(seq 181 1 189) [1900]=$(seq 175 1 181) [2000]=$(seq 160 1 170) [2100]=$(seq 156 1 166) [2200]=$(seq 153 1 161) [2300]=$(seq 144 1 152) [2400]=$(seq 138 1 148) [2500]=$(seq 133 1 143))
#declare -A LLPT_Tdict=([1700]=$(seq 193 1 196) [1800]=$(seq 184 1 184) [1900]=$(seq 176 1 177) [2000]=$(seq 169 1 171) [2100]=$(seq 161 1 163) [2200]=$(seq 154 1 157) [2300]=$(seq 148 1 150) [2400]=$(seq 142 1 145) [2500]=$(seq 137 1 140))
#declare -A LLPT_Tdict=([1700]=$(seq 193 1 195) [1800]=$(seq 184 1 186) [1900]=$(seq 176 1 177) [2000]=$(seq 168 1 171) [2100]=$(seq 161 1 163) [2200]=$(seq 155 1 157) [2300]=$(seq 148 1 150) [2400]=$(seq 142 1 145) [2500]=$(seq 137 1 140))
#declare -A LLPT_Tdict=([1700]=$(seq 191.5 1 194.5) [1800]=$(seq 183 0.2 185) [1900]=$(seq 176.2 0.2 177.8) [2000]=$(seq 168 0.2 169.8) [2100]=$(seq 160.2 0.2 164.8) [2200]=$(seq 154.2 0.2 156.8) [2300]=$(seq 148.2 0.2 149.8) [2400]=$(seq 142.5 0.5 145.5))
#declare -A LLPT_Tdict=([1800]="170 190 192 194" [1900]="160 182 184 186" [2000]=$(seq 172 2 184) [2100]=$(seq 170 2 180) [2200]=$(seq 162 10 172) [2300]=$(seq 162 10 172) [2400]=$(seq 150 10 160) [2500]=$(seq 150 10 160))

declare -A LLPT_Tdict=([1700]=$(seq 192 0.5 194) [1800]=$(seq 183 0.2 185) [2000]=$(seq 167.4 0.2 168.8) [2100]=$(seq 160 0.2 162) [2300]=$(seq 146.2 0.2 149.4) [2500]=$(seq 134 0.5 138) [3000]=$(seq 96 2 126))
# declare -A leftover_Tdict=([1700]="192.5 193 193.5" [1800]="$(seq 183.2 0.2 183.8) $(seq 184.2 0.2 184.8)" [2000]="$(seq 167.4 0.2 167.8) $(seq 168.2 0.2 168.8)" [2100]="$(seq 160.2 0.2 160.8) $(seq 161.2 0.2 161.8)" \
# [2300]="$(seq 146.2 0.2 146.8) $(seq 147.2 0.2 147.8) $(seq 148.2 0.2 148.8) $(seq 149.2 0.2 149.4)" [2500]="135 135.5 136.5 137 137.5")
#declare -A LLPT_Tdict=([1700]="186 188 190 196 198" [1800]="180 182 186 188" [2000]=$(seq 162 2 166 ; seq 170 2 174) [2100]="156 158 163 164 166" \
# [2300]="142 144 150 152" [2500]="130 132 140 142")
#for temp in $(seq 1700 100 2400)
#for temp in 1700 1800 2000 2100 2300 2500 3000
for temp in 1700
do
    for ph in ${phases[@]}
    do 
        case=NPT_mliap
        #for p in ${leftover_Tdict[$temp]}
        #for p in ${LLPT_Tdict[$temp]}
        #for p in 192.25 192.75 193.25 193.75
        for p in 194.5 195
        #for p in 172
        do
            #temp=1800
            if [[ $p == *".0" ]]; then
				p=${p%.*}
			fi
            python ./lammps_out_reader_phases.py -p $p -t $temp -f $ph -c $case
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
                -J 1200_bragg_${p}_${temp}_${ph} \
                --mem=10g \
                --mail-type=NONE \
                ./run_braggs.sh | tr -cd "[0-9]")
        done
    done
done
