module PDCSGPUExt

using CUDA
using PDCS

include(joinpath(pkgdir(PDCS), "src", "pdcs_gpu", "PDCS_GPU.jl"))

function __init__()
    PDCS._register_gpu!(PDCS_GPU)
    return
end

end
