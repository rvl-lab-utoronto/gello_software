from gello.data_utils.save_data_to_hdf5 import gather_demonstrations_as_hdf5



def main():
    in_dir = "/home/ruthrash/gello_data/gello"
    gather_demonstrations_as_hdf5(in_dir=in_dir, out_dir="/home/ruthrash/gello_data")
 

if __name__ == "__main__":
    main()