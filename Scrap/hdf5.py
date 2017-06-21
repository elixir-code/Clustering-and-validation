def sample_array():
	import h5py

	f = h5py.File("sample.hdf5","w")	
	dset = f.create_dataset("mydataset",(1000,1000),dtype='d')

	return dset