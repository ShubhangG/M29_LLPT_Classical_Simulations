from ase import io
import sys
sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/mysharelib")
sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/solid_hydrogen")
sys.path.append("/work/nvme/bcqo/shubhanggoswami/paul_ricky_codes/harvest_qmcpack")
from qharv_db.ase_md import mhcpc_supercell
from mytool import myio
from mytool import myfun

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--pgpa', type=int, default=50,required=True)
    parser.add_argument('--natom', type=int, default=360, required=True)
    args= parser.parse_args()
    pgpa = args.pgpa
    natom= args.natom

    atoms=mhcpc_supercell(pgpa,natom)
    myfun.mkdir("p{}".format(pgpa))
    myio.atoms2data(atoms,"p{}/data.txt".format(pgpa),["H"])
    myio.atoms2ipixyz(atoms,"p{}/init.xyz".format(pgpa))

if __name__ == '__main__':
    main()
