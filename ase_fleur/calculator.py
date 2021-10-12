# -*- coding: utf-8 -*-
from pathlib import Path
import warnings
import re

from masci_tools.io.fleurxmlmodifier import FleurXMLModifier
from masci_tools.io.parsers.fleur import outxml_parser

from ase.calculators.genericfileio import GenericFileIOCalculator, CalculatorTemplate
from ase.io import write


class FleurProfile:
    def __init__(self, argv, inpgen_argv):
        self.argv = argv
        self.inpgen_argv = inpgen_argv

    def version(self) -> str:
        """
        Return the version string of the fleur code in this profile
        """
        from subprocess import check_output
        import tempfile

        with tempfile.TemporaryFile('w') as err:
            out = check_output(self.argv + ["-info"], stderr=err).decode('utf-8')
        m = re.findall(r'^(.*)\(www\.max\-centre\.eu\)',out,flags=re.MULTILINE)
        if not m:
            raise ValueError(f"Could not retrieve version from output: {out}")
        return m[0].strip()

    def run(self, directory, outputfile, error_file):
        from subprocess import check_call

        with open(outputfile, "w") as fd:
            with open(error_file, "w") as ferr:
                check_call(self.argv, stdout=fd, stderr=ferr, cwd=directory)

    def run_inpgen(self, directory, inputfile, outputfile, error_file):
        from subprocess import check_call

        with open(outputfile, "w") as fd:
            with open(error_file, "w") as ferr:
                check_call(self.inpgen_argv + ["-f", str(inputfile)], stdout=fd, stderr=ferr, cwd=directory)


class FleurTemplate(CalculatorTemplate):
    def __init__(self, *, inpgen_profile):
        super().__init__(name="fleur", implemented_properties=("energy"))
        self.output_file = "fleur.log"
        self.error_file = "error.log"
        self.inpgen_profile = inpgen_profile
        self.max_runs = 3
        self.iter_per_run = 30
        self.distance_converged = 1e-6

    def write_input(self, directory, atoms, parameters, properties):
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
                warnings.warn("inpgen or inputgenerator has to appear in the inpgen file title" "Added to the end")
                parameters["title"] += " (inpgen)"

        inputfile = directory / "fleur.in"
        write(inputfile, atoms, parameters=parameters, format='fleur-inpgen')

        # 2. Run inpgen
        self.execute_inpgen(directory, self.inpgen_profile, inputfile)

        # 3. ggf. make modifications using the FleurXMLmodifier
        if inp_changes:
            fm = FleurXMLModifier()
            fm.set_inpchanges({"itmax": self.iter_per_run})
            fm.add_task_list(inp_changes)
            xmltree, _ = fm.modify_xmlfile(directory / "inp.xml")
            xmltree.write(directory / self.input_file, encoding="utf-8", pretty_print=True)

    def execute(self, directory, profile) -> None:
        converged = False
        run = 1
        while not converged and run <= self.max_runs:
            profile.run(directory, self.output_file, self.error_file)

            fleur_results = outxml_parser(directory / "out.xml")

            if "overall_density_convergence" in fleur_results:
                distance = fleur_results["overall_density_convergence"]
            else:
                distance = fleur_results.get("density_convergence")

            converged = distance < self.distance_converged

    def execute_inpgen(self, directory, profile, inputfile) -> None:
        profile.run_inpgen(directory, inputfile, self.output_file, self.error_file)

    def read_results(self, directory):
        fleur_results = outxml_parser(directory / "out.xml")

        if "overall_density_convergence" in fleur_results:
            distance = fleur_results["overall_density_convergence"]
        else:
            distance = fleur_results.get("density_convergence")

        if not distance < self.distance_converged:
            raise RuntimeError("Fleur calculation did not converge")

        return fleur_results


class Fleur(GenericFileIOCalculator):
    def __init__(self, *, profile=None, directory=".", **kwargs):

        if profile is None:
            profile = FleurProfile(["fleur"], ["inpgen"])

        super().__init__(
            template=FleurTemplate(inpgen_profile=profile), profile=profile, directory=directory, parameters=kwargs
        )
