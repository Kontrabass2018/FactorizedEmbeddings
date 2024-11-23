using Pkg 
Pkg.activate(".")
# Pkg.add("cuDNN")
# Pkg.add("CUDA")
# Pkg.add("HDF5")
# Pkg.add("CSV")
using cuDNN
using CUDA
#data related
using HDF5
using CSV 
using Random
function load_tcga_data(infilename; shfl = true)
    infile = h5open(infilename)
    TCGA_data = infile["data"][:,:]
    labs = string.(infile["labels"][:])
    samples = string.(infile["samples"][:])
    genes = string.(infile["genes"][:]) 
    biotypes = string.(infile["biotypes"][:])
    close(infile)
    ids = collect(1:size(labs)[1])
    shfl && (ids = shuffle(ids))
    return TCGA_data[ids,:], labs[ids], samples[ids], genes, biotypes
end 

function fetch_data(filename; shfl = true)
    if !(filename in readdir("."))
        # Define the URL`
        tcga_data_url = "https://bioinfo.iric.ca/~sauves/VARIA/$filename"

        # Escape the URL to handle special characters
        escaped_url = Base.shell_escape(tcga_data_url)

        # Construct and execute the wget command
        command = `wget $escaped_url`
        run(command) 
    end 
    load_tcga_data(filename; shfl = shfl)

end 