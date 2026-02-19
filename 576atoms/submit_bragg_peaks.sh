phases=(liquid)
source /projects/bcqo/shubhanggoswami/mace_physice/bin/activate
cd /work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/576atoms
declare -A forcefiller_dict=([1700]="195 197" [1800]="179 189" [2000]="171 173 175 165" [2100]="157 167 165" [2300]="145 153 155" [2500]="131 133 141 139")

for temp in 1700 1800 2000 2100 2300 2500
do
    for ph in ${phases[@]}
    do 
        case=NPT_mliap
        for p in ${forcefiller_dict[$temp]}
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
                -J 576_bragg_${p}_${temp}_${ph} \
                --mem=10g \
                --mail-type=NONE \
                ./run_braggs.sh | tr -cd "[0-9]")
        done
    done
done
