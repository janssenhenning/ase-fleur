# -*- coding: utf-8 -*-
"""
Tests of the fleur calculator class
"""
from ase.build import bulk
import pytest
import numpy as np
from packaging.version import Version


def verify(calc):
    assert calc.get_fermi_level() is not None
    assert calc.get_ibz_k_points() is not None
    assert calc.get_eigenvalues(spin=0, kpt=0) is not None
    assert calc.get_number_of_spins() is not None
    assert calc.get_k_point_weights() is not None


@pytest.mark.calculator("fleur")
def test_main(factory):
    """
    Basic test of Fleur calculator
    """
    atoms = bulk("Si")
    atoms.calc = factory.calc()
    atoms.get_potential_energy()
    verify(atoms.calc)


@pytest.mark.calculator("fleur")
def test_force(factory):
    """
    Basic test of Fleur calculator
    """
    atoms = bulk("Si")
    atoms.calc = factory.calc()
    assert atoms.get_forces() == pytest.approx(np.array([[0, 0, 0], [0, 0, 0]]))
    verify(atoms.calc)


@pytest.mark.calculator("fleur")
def test_version(factory):
    """
    Test of version parsing
    """
    code_version = factory.factory.version()
    code_version = Version(code_version)  # Ensures that the parsed version has a valid format

    assert (
        code_version.major >= 6
    )  # This of course assumes that we have atleast version 6.0 (earliest version on conda-forge)
    assert code_version.micro == 0  # Up till now the version of the fleur code only has major and minor version
