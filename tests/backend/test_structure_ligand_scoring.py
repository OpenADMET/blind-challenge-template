"""Real OpenStructure scoring of synthetic complexes: one model ligand only.

Skipped when OpenStructure, its compound library or PoseBusters isn't installed
(``score_single_structure`` needs PoseBusters after OST scoring). The scoring Lambda
image and ``environment.yml`` both include all three.
"""

import math

import pytest

from backend.evaluate_predictions import score_single_structure
from tests.backend.synthetic_complexes import (
    OFFSET,
    SITE,
    SITE_2,
    benzene,
    complex_pdb,
    cyclobutane,
)

pytest.importorskip("posebusters")
ost_conop = pytest.importorskip("ost.conop")
if ost_conop.GetDefaultLib() is None:
    pytest.skip("OpenStructure compound library not available", allow_module_level=True)


def _score_and_checks(
    tmp_path,
    model_pdb: str,
    ref_pdb: str,
    smiles: str | None = None,
    max_pb_failures: int = 99,
) -> tuple[dict[str, float], list[str]]:
    model = tmp_path / "model.pdb"
    ref = tmp_path / "ref.pdb"
    model.write_text(model_pdb)
    ref.write_text(ref_pdb)
    return score_single_structure(
        str(model), str(ref), max_pb_failures=max_pb_failures, smiles=smiles
    )


def _score(
    tmp_path, model_pdb: str, ref_pdb: str, smiles: str | None = None
) -> dict[str, float]:
    return _score_and_checks(tmp_path, model_pdb, ref_pdb, smiles)[0]


def test_single_ligand_scores(tmp_path):
    ref = complex_pdb(benzene(SITE))
    assert _score(tmp_path, ref, ref)["LDDT-PLI"] == pytest.approx(1.0)
    off = _score(tmp_path, complex_pdb(benzene(OFFSET)), ref)
    assert off["LDDT-PLI"] < 1.0
    assert off["BiSyRMSD"] == pytest.approx(3.0, abs=0.01)


def test_model_with_two_ligands_is_not_best_of_two(tmp_path):
    """A wrong pose plus the right pose must not score as the right pose."""
    ref = complex_pdb(benzene(SITE))
    model = complex_pdb(
        benzene(OFFSET, "B", 1), benzene(SITE, "C", 1, first_serial=600)
    )
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


# --- Ligand bonds from SMILES (#27) -------------------------------------------


def test_strained_ring_scores_with_smiles(tmp_path):
    """A false bond across a strained ring in the reference made the compound
    unscoreable; with the SMILES both ligands get the same, correct bonds."""
    ref = complex_pdb(cyclobutane(SITE, diagonal=1.95))  # false C1-C3 bond
    model = complex_pdb(cyclobutane(SITE, diagonal=2.15))  # no false bond

    without = _score(tmp_path, model, ref)
    assert all(math.isnan(v) for v in without.values())

    with_smiles = _score(tmp_path, model, ref, smiles="C1CCC1")
    assert with_smiles["LDDT-PLI"] == pytest.approx(1.0)
    assert with_smiles["BiSyRMSD"] < 0.2


@pytest.mark.parametrize("model_site", [SITE, OFFSET])
def test_smiles_leaves_ordinary_ligand_scores_unchanged(tmp_path, model_site):
    ref = complex_pdb(benzene(SITE))
    model = complex_pdb(benzene(model_site))
    without = _score(tmp_path, model, ref)
    with_smiles = _score(tmp_path, model, ref, smiles="c1ccccc1")
    assert with_smiles == pytest.approx(without)


def test_wrong_smiles_falls_back_to_distance_bonds(tmp_path):
    ref = complex_pdb(benzene(SITE))
    model = complex_pdb(benzene(OFFSET))
    assert _score(tmp_path, model, ref, smiles="c1ccncc1") == pytest.approx(
        _score(tmp_path, model, ref)
    )


def test_reference_with_two_copies_matches_either_with_smiles(tmp_path):
    ref = complex_pdb(benzene(SITE, "B", 1), benzene(SITE_2, "C", 1, first_serial=600))
    for site in (SITE, SITE_2):
        scores = _score(tmp_path, complex_pdb(benzene(site)), ref, smiles="c1ccccc1")
        assert scores["LDDT-PLI"] == pytest.approx(1.0)
        assert scores["BiSyRMSD"] == pytest.approx(0.0, abs=1e-3)


def test_smiles_bond_orders_stop_false_posebusters_failures(tmp_path):
    """Without bond orders, PoseBusters reads benzene as a saturated ring and
    fails a flatness check, which zeroes a plausible pose."""
    ref = complex_pdb(benzene(SITE))
    model = complex_pdb(benzene(OFFSET))
    _, failed_without = _score_and_checks(tmp_path, model, ref, max_pb_failures=0)
    assert failed_without
    scores, failed_with = _score_and_checks(
        tmp_path, model, ref, smiles="c1ccccc1", max_pb_failures=0
    )
    assert failed_with == []
    assert scores["LDDT-PLI"] > 0
