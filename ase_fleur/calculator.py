# -*- coding: utf-8 -*-
"""
This module defines a calculator for the Fleur code starting from version v27
"""
from __future__ import annotations

from pathlib import Path
import warnings
import re
from typing import Any

from masci_tools.io.fleurxmlmodifier import FleurXMLModifier
from masci_tools.io.parsers.fleur import outxml_parser

from ase.calculators.genericfileio import GenericFileIOCalculator, CalculatorTemplate, BaseProfile
from ase import Atoms

from ase_fleur.io import write_fleur_inpgen, read_fleur_outxml


def _parse_fleur_max_version_from_output(base_argv: list[str]) -> str:
    """
    Parses the MaX version for the provided fleur/inpgen code

    Note that the passed command must not contain the -version flag
    """
    from subprocess import check_output
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = check_output([*base_argv, "-version"], cwd=td, encoding="utf-8")
    m = re.findall(r"^\s*MaX\-Release (.*)\(www\.max\-centre\.eu\)", out, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Could not retrieve version from output: {out}")
    return m[0].strip()


class InpgenProfile(BaseProfile):  # type: ignore[misc]
    """
    Profile for executing the Fleur input generator code

    :param command: arguments for the Fleur code
    :param flags: list of arguments/flags to be added to the execution
    """

    def __init__(self, command, flags=None):
        super().__init__(command)
        if flags is None:
            flags = []
        self.flags = flags

    def version(self) -> str:
        """
        Return the version string of the fleur code in this profile
        """
        return _parse_fleur_max_version_from_output(self._split_command)

    def get_calculator_command(self, inputfile):
        return ["-f", str(inputfile), *self.flags]


class FleurProfile(BaseProfile):  # type: ignore[misc]
    """
    Profile for executing the Fleur code

    :param command: arguments for the Fleur code
    :param flags: list of arguments/flags to be added to the execution
    """

    def __init__(self, command, flags=None):
        super().__init__(command)
        if flags is None:
            flags = []
        self.flags = flags

    def version(self) -> str:
        """
        Return the version string of the fleur code in this profile
        """
        return _parse_fleur_max_version_from_output(self._split_command)

    def get_calculator_command(self, inputfile):
        return self.flags


class FleurTemplate(CalculatorTemplate):  # type: ignore[misc]
    """
    Template defining a Fleur Calculation
    """

    def __init__(self, *, inpgen_profile: InpgenProfile) -> None:
        super().__init__(
            name="fleur",
            implemented_properties=("energy", "forces", "magmom", "magmoms", "efermi", "free_energy", "charges"),
        )
        self.stdout_file = "fleur.log"
        self.error_file = "error.log"
        self.output_file = "out.xml"
        self.inpgen_profile = inpgen_profile
        self.max_runs = 3
        self.iter_per_run = 30
        self.density_converged = 1e-6
        self.force_convergence = {"force_converged": 0.002, "qfix": 2, "forcealpha": 1.0, "forcemix": "straight"}

    def write_input(  # pylint: disable=too-many-positional-arguments
        self,
        profile: FleurProfile,
        directory: Path,
        atoms: Atoms,
        parameters: dict[str, Any],
        properties: list[str],
    ) -> None:
        """
        Create Fleur inp.xml file from atoms object by calling the
        Fleur inpgen

        :param directory: path to the calculation directory
        :param atoms: ase.Atoms object to use
        :param parameters: Dict with inpgen parameters
                           Changes to be done after the inpgen was run
                           can be specified in the entry inpxml_changes
        """

        # Sketch
        # 1. Create inpgen input using the fleur IO format
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        parameters = dict(parameters)
        inp_changes = parameters.pop("inpxml_changes", [])
        if "title" not in parameters:
            parameters["title"] = "Fleur inpgen input generated from ASE"
        else:
            if all(s not in parameters["title"] for s in ("inpgen", "input generator")):
                warnings.warn("inpgen or inputgenerator has to appear in the inpgen file title. Added to the end")
                parameters["title"] += " (inpgen)"

        inputfile = directory / "fleur.in"
        write_fleur_inpgen(inputfile, atoms, parameters=parameters)

        # 2. Run inpgen
        self.execute_inpgen(directory, inputfile)

        # 3. Modify inp.xml according to set parameters
        fm = FleurXMLModifier()
        fm.set_inpchanges({"itmax": self.iter_per_run, "mindistance": self.density_converged})

        if "forces" in properties:
            fm.set_inpchanges(
                {
                    "force_converged": self.force_convergence["force_converged"],
                    "l_f": True,
                    "qfix": self.force_convergence["qfix"],
                    "forcealpha": self.force_convergence["forcealpha"],
                    "forcemix": self.force_convergence["forcemix"],
                }
            )

        # 4. ggf. make custom modifications using the FleurXMLmodifier
        if inp_changes:
            fm.add_task_list(inp_changes)

        xmltree, _ = fm.modify_xmlfile(directory / "inp.xml")
        xmltree.write(directory / "inp.xml", encoding="utf-8", pretty_print=True)

    def execute(self, directory: Path, profile: FleurProfile) -> None:
        """
        Execute Fleur multiple times until the calculation is either converged
        or a maximum number of iterations is reached

        :param directory: Path to the calculation directory
        :param profile: FleurProfile to use
        """
        converged = False
        run = 1
        while not converged and run <= self.max_runs:
            profile.run(directory, self.stdout_file, self.error_file)
            converged = self.check_convergence(directory)

    def execute_inpgen(self, directory: Path, inputfile: Path) -> None:
        """
        Execute Fleur inpgen to create the Fleur inp.xml

        :param directory: Path to the calculation directory
        :param profile: FleurProfile to use
        :param inputfile: Path to the inputfile to use
        """
        self.inpgen_profile.run(directory, inputfile, self.stdout_file, self.error_file)

    def check_convergence(self, directory: Path) -> bool:
        """
        Check if the calculation is converged

        :param directory: Path to the calculation directory
        """
        fleur_results = outxml_parser(directory / self.output_file, iteration_to_parse="all")

        MAGNETIC_DISTANCE_KEY = "overall_density_convergence"
        DISTANCE_KEY = "density_convergence"

        distance = fleur_results.get(MAGNETIC_DISTANCE_KEY, fleur_results.get(DISTANCE_KEY))
        if distance is None:
            raise ValueError("Could not find charge density distance in output file")
        if fleur_results["fleur_modes"]["relax"]:
            distance = distance[-2]
        else:
            distance = distance[-1]

        return distance < self.density_converged

    def read_results(self, directory: Path) -> dict[str, Any]:
        """
        Read the calculation results from the produced out.xml

        :param directory: Path to the calculation directory
        """
        atoms = read_fleur_outxml(directory / self.output_file)
        return dict(atoms.calc.properties())

    def load_profile(self, cfg, **kwargs):
        return FleurProfile.from_config(cfg, self.name, **kwargs)


class Fleur(GenericFileIOCalculator):  # type: ignore[misc]
    """
    Ase Calculator for FLEUR calculations
    """

    def __init__(
        self,
        *,
        profile: FleurProfile | None = None,
        inpgen_profile: InpgenProfile | None = None,
        directory: str | Path = ".",
        **kwargs: Any,
    ) -> None:
        if profile is None:
            profile = FleurProfile(
                ["fleur"],
            )
        if inpgen_profile is None:
            inpgen_profile = InpgenProfile(
                ["inpgen"],
            )

        super().__init__(
            template=FleurTemplate(inpgen_profile=inpgen_profile),
            profile=profile,
            directory=directory,
            parameters=kwargs,
        )
