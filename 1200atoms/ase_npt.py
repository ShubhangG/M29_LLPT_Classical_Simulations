import time
from datetime import datetime
start_time = time.time()
start_human_time = datetime.now() 
import ase.units as units
from ase.io import read
from ase.md import MDLogger
from ase.md.nptberendsen import NPTBerendsen
from ase.io.trajectory import Trajectory
from mace.calculators import MACECalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='start ASE NPT run with MACE')
    parser.add_argument('-r','--run_num',dest='run_num',type=int,help='run number',default=0)
    parser.add_argument('-p','--press',dest='pressure',type=float,help='pressure in GPa',required=True)
    parser.add_argument('-t','--temp',dest='temp',type=int,help='temperature in K',required=True)
    parser.add_argument('-c','--case',dest='case',type=str,help="Is it NVT or NPT simulation",default="NPT")
    parser.add_argument('-nsteps',dest='nsteps',type=int,help='number of timesteps',default=10000)
    args= parser.parse_args()
    run_num= args.run_num
    pressure= args.pressure * 10000  # Convert GPa to bar
    temperature= args.temp
    run_type = args.case
    number_of_timesteps = args.nsteps

    model = "/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/calculator/M29_elite_LLPT-07-29-2025.model"

    if pressure.is_integer():
        pressure = int(pressure)
    
    cueq_flag = True 
    Dispersion_flag = False 
    device_flag = 'cuda'

    #number_of_timesteps = 50000
    #temperature = 2100.0
    timestep_fs = 0.50
    #pressure = 1620000     # Desired pressure in bar
    taut_a   = 10          # Time constant for Temperature coupling 
    taup_a = 100           # Time constant for Pressure coupling
    compressibility_a = 4.57e-5 # water compressibility at normal conditions 4.57e-5 
    #compressibility_a = 1.0e-5 #roughly obtained from EOS_fits
    velocity_rescaling = False 
    saving_interval = 10

    #run_type = 'NPT'
    #run_num = 1

    if run_num == 0:
        init_conf = "../../../1300K_data.txt"
        init_conf = read(f'{init_conf}', index = '-1', format = 'lammps-data')
    else:
        init_conf = f"./trajectory_{run_type}_{run_num-1}.traj"
        init_conf = read(f'{init_conf}', index = '-1', format = 'traj')



    calculator = MACECalculator(model_paths = f'{model}', device = device_flag, default_dtype = 'float64', dispersion = Dispersion_flag, enable_cueq = cueq_flag)
    init_conf.calc = calculator

    with open(f'./report_readme_{run_type}_{run_num}.txt', "w") as my_file:
        my_file.write(f"NPT Berendsen\n")
        my_file.write(f"Model name: {model}\n")
        my_file.write(f"Temperature: {temperature:.2f} K\n")
        my_file.write(f"Pressure: {pressure:.4f} [bar]\n")
        my_file.write(f"Compressibility au: {compressibility_a:.4e} [bar^-1]\n")
        my_file.write(f"Taut: {taut_a:.4f} ts \n")
        my_file.write(f"Taup: {taup_a:.4f} ts\n")
        my_file.write(f"Timestep: {timestep_fs:.3f} fs\n")
        my_file.write(f"Number of timesteps: {number_of_timesteps} \n")
        my_file.write(f"Start time: {start_human_time}\n")

    if (velocity_rescaling == True):
        MaxwellBoltzmannDistribution(init_conf, temperature_K = temperature, force_temp = True)

    dyn = NPTBerendsen(init_conf,
            timestep = timestep_fs * units.fs,
            temperature_K = temperature,
            taut = taut_a * units.fs,
            taup = taup_a * units.fs,
            pressure_au = pressure * units.bar,
            compressibility_au = compressibility_a / units.bar  
        )

    dyn.attach(MDLogger(dyn, init_conf, f'./md_{temperature}K_{pressure/10000}GPa_{run_type}_{run_num}.log', header = True, stress = True, peratom = True, mode = "w"), interval = 1)

    traj = Trajectory(f'./trajectory_{run_type}_{run_num}.traj', 'w', init_conf)
    dyn.attach(traj.write, interval = saving_interval)

    start_dyn_time = time.time()
    dyn.run(number_of_timesteps) 

    end_dyn_time = time.time()
    end_human_time = datetime.now()
    net_run_time = end_dyn_time - start_dyn_time
    total_time =  end_dyn_time - start_time
    time_per_timestep = net_run_time / number_of_timesteps

    total_time_h = total_time // 3600
    remaining_seconds = total_time % 3600
    total_time_min = remaining_seconds // 60
    total_time_sec = remaining_seconds % 60

    with open(f'./report_readme_{run_type}_{run_num}.txt', "a") as my_file:
        
        my_file.write("\nDynamics ended successfully!\n")
        my_file.write(f"End time: {end_human_time}\n")
        my_file.write(f"Net run time: {net_run_time:.2f} s\n")
        my_file.write(f"Total time: {total_time:.2f} s\n")
        my_file.write(f"Total time: {int(total_time_h)} h {int(total_time_min)} m {int(total_time_sec)} s \n")
        my_file.write(f"Time per timestep: {time_per_timestep:.2f} s\n")
