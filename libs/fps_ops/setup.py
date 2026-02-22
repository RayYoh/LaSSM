import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

setup(
    name='fps_ops',
    packages=["fps_ops"],
    package_dir={"fps_ops": "functions"},
    ext_modules=[
        CUDAExtension(
            name='fps_ops_cuda',
            sources=[
                'src/sampling.cpp',
                'src/sampling_gpu.cu'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
