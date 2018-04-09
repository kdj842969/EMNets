import h5py
import mrcfile as mrc

with h5py.File('reconstruction.hdf5', 'r') as f:
    decoded_imgs = f['recon_mrc'][...]

print(decoded_imgs.shape)
example = decoded_imgs[1]

def save_mrc(decoded_imgs=None, n=1):
    path = "./reconstruction/"
    for i in range(n):
        if decoded_imgs is not None:
            file_name = path + "mrc" + str(i) + ".mrc"
            newmrc = mrc.new(file_name, overwrite=True)
            newmrc.set_data(decoded_imgs[i])
            newmrc.update_header_stats()
            newmrc.close()


save_mrc(decoded_imgs, 14)