
for temp in 1700 
do
    cd p200/liquid/NPT/$temp/ 
    Tlast=$(awk '/^ITEM: TIMESTEP/{getline; t=$1} END{print t}' dump.init.atom)
    echo "Last dump step = $Tlast"
    JOB=$(sbatch \
        -p gpuA40x4 \
        --gpus=1 \
        -N 1 \
        --mem=10g \
        --ntasks=16 \
        --account=bcqo-delta-gpu \
        -t 01:15:00 \
        --export=ALL,Tlast=$Tlast \
        -J 576atom_M29_200_${temp}_NPT_recovery \
        --mail-type=NONE \
        ../../../../slurm_recovery.sh | tr -cd "[0-9]")

    cd ../../../..
done