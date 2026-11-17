"""Real OpenStructure scoring of synthetic complexes: one model ligand only.

Skipped when OpenStructure, its compound library or PoseBusters isn't installed.
The scoring Lambda image has all three; ``environment.yml`` (used by CI) doesn't
include PoseBusters yet, and ``score_single_structure`` needs it after OST scoring.
"""

import math

import pytest

from backend.evaluate_predictions import score_single_structure
from tests.backend.synthetic_complexes import OFFSET, SITE, SITE_2, benzene, complex_pdb

pytest.importorskip("posebusters")
ost_conop = pytest.importorskip("ost.conop")
if ost_conop.GetDefaultLib() is None:
    pytest.skip("OpenStructure compound library not available", allow_module_level=True)


def _score(tmp_path, model_pdb: str, ref_pdb: str) -> dict[str, float]:
    model = tmp_path / "model.pdb"
    ref = tmp_path / "ref.pdb"
    model.write_text(model_pdb)
    ref.write_text(ref_pdb)
    scores, _ = score_single_structure(str(model), str(ref), max_pb_failures=99)
    return scores


def test_single_ligand_scores(tmp_path):
    ref = complex_pdb(benzene(SITE))
    assert _score(tmp_path, ref, ref)["LDDT-PLI"] == pytest.approx(1.0)
    off = _score(tmp_path, complex_pdb(benzene(OFFSET)), ref)
    assert off["LDDT-PLI"] < 1.0
    assert off["BiSyRMSD"] == pytest.approx(3.0, abs=0.01)


def test_model_with_two_ligands_is_not_best_of_two(tmp_path):
    """A wrong pose plus the right pose must not score as the right pose."""
    ref = complex_pdb(benzene(SITE))
    model = complex_pdb(benzene(OFFSET, "B", 1), benzene(SITE, "C", 1, first_serial=600))
    scores = _score(tmp_path, model, ref)
    assert all(math.isnan(v) for v in scores.values())


def test_reference_with_two_copies_matches_either(tmp_path):
    """The reference may hold several ligand copies; the one model ligand is scored
    against each and the best match is kept."""
    ref = complex_pdb(benzene(SITE, "B", 1), benzene(SITE_2, "C", 1, first_serial=600))
    for site in (SITE, SITE_2):
        scores = _score(tmp_path, complex_pdb(benzene(site)), ref)
        assert scores["LDDT-PLI"] == pytest.approx(1.0)
        assert scores["BiSyRMSD"] == pytest.approx(0.0, abs=1e-3)


def test_altloc_conformers_score_altloc_a(tmp_path):
    ref = complex_pdb(benzene(SITE))
    model = complex_pdb(
        benzene(SITE, "B", 1, altloc="A"),
        benzene(OFFSET, "B", 1, altloc="B", first_serial=600),
    )
    assert _score(tmp_path, model, ref)["BiSyRMSD"] == pytest.approx(0.0, abs=1e-3)
