from ase import io
import numpy as np
from natsort import natsorted
import os
import argparse

def gen_data_path(P,T,phase,case):
    return f"/work/hdd/bcqo/shubhanggoswami/LLPT_Scan/M29/576atoms/p{P}/{phase}/{case}/{T}/"

def merge_and_dump_lammps(P,T,phase,case):
    all_data = natsorted(os.listdir(gen_data_path(P,T,phase,case)))
    filtering_substring = ["lammps.out"]
    simout_filelist = list(filter(lambda x: any(substring in x for substring in filtering_substring), all_data))
    if any('lammps.out.init' in x for x in simout_filelist):
        simout_filelist.remove("lammps.out.init.txt")
        simout_filelist.insert(0,"lammps.out.init.txt")
    if any('lammps.recovery.out' in x for x in simout_filelist):
        simout_filelist.remove("lammps.recovery.out.txt")
        simout_filelist.insert(1,"lammps.recovery.out.txt")
    #print(simout_filelist)
    FO = open(gen_data_path(P,T,phase,case)+f"analysis/P{P}T{T}M29_merged_lammps_out.csv",'w')
    header_=[]
    sim_out_array=[]
    FO_header_write_count = False
    for file_name in simout_filelist:
        with open(gen_data_path(P,T,phase,case)+file_name) as fin:
            all_rows = fin.read().split('\n')
            for idx,row in enumerate(all_rows):
                columns = row.split()
                # if columns[0]=='Step':
                #     print(columns)
                #     print(len(columns))
                if len(columns)==22:
                    # print(columns)
                    # break
                    if columns[0]=='Step':
                        header_=columns
                        if not FO_header_write_count:
                            FO.write(','.join(columns))
                            FO.write('\n')
                        FO_header_write_count=True
                        continue
                    FO.write(','.join(columns))
                    FO.write('\n')
                    sim_out_array.append(list(map(float,columns)))
    FO.close()
    sim_out_array=np.array(sim_out_array)
    np.save(gen_data_path(P,T,phase,case)+f"analysis/P{P}T{T}M29_merged_lammps_out",sim_out_array)


def main():
    parser = argparse.ArgumentParser(description='input P-T to merge lammps files')
    parser.add_argument('-p','--press',dest='pressure',type=float,help='pressure',required=True)
    parser.add_argument('-t','--temp',dest='temp',type=int,help='temperature',required=True)
    parser.add_argument('-c','--case',dest='case',type=str,help='NPT or NVT',default="NPT")
    parser.add_argument('-f','--phase',dest='phase',type=str,help='solid or liquid',default="liquid")
    
    args= parser.parse_args()
    press= args.pressure
    tkel = args.temp
    case = args.case
    phase = args.phase
    if press.is_integer():
        press = int(press)
    
    merge_and_dump_lammps(press,tkel,phase,case)

    # for press in range(50,101,25):
    #     for tkel in range(1300,2001,100):
    #         merge_and_dump_lammps(press,tkel,"M14")

    # for tkel in range(1000,2001,100):
    #     merge_and_dump_lammps(150,tkel,"M14")


if __name__ == '__main__':
    main()


