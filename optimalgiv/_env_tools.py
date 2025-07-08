"""Environment and package update tools"""

from juliacall import Main as jl


def update_packages():
    """Update Julia packages in the environment"""
    jl.seval("import Pkg")
    jl.seval("Pkg.update()")
    print("Julia packages updated successfully.")