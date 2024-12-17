import h5py

with h5py.File("data.h5", "r") as f:
    for key in f["flight_data"]:
        print(f"{key}: {f['flight_data'][key][:]}")








