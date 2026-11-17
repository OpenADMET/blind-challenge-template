"""Tests for taking ligand bonds from the ground-truth SMILES (#27)."""

import pytest

from tests.backend.synthetic_complexes import (
    SITE,
    SITE_2,
    benzene,
    complex_pdb,
    cyclobutane,
)

Chem = pytest.importorskip("rdkit.Chem")
AllChem = pytest.importorskip("rdkit.Chem.AllChem")

from backend.ligand_bonds import (  # noqa: E402
    assign_bond_orders_from_template,
    ligands_from_pdb,
)


def _pdb_ligand(pdb_text: str) -> "Chem.Mol":
    """The LIG atoms of a PDB block, read as RDKit reads a submission."""
    mol = Chem.MolFromPDBBlock(pdb_text, removeHs=False, sanitize=False)
    lig = Chem.RWMol(mol)
    for atom in sorted(mol.GetAtoms(), key=lambda a: -a.GetIdx()):
        if atom.GetPDBResidueInfo().GetResidueName() != "LIG":
            lig.RemoveAtom(atom.GetIdx())
    return lig.GetMol()


def _embedded_pdb_ligand(smiles: str, with_hydrogens: bool) -> "Chem.Mol":
    """A 3D ligand written to PDB as LIG and read back with guessed bonds."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    if not with_hydrogens:
        mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("LIG")
        info.SetChainId("B")
        info.SetResidueNumber(1)
        info.SetIsHeteroAtom(True)
        info.SetName(f"{atom.GetSymbol()}{atom.GetIdx()}".ljust(4)[:4])
        atom.SetMonomerInfo(info)
    return _pdb_ligand(Chem.MolToPDBBlock(mol, flavor=2))  # flavor 2: no CONECT


def _canonical(mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def test_false_ring_bond_is_removed():
    ligand = _pdb_ligand(complex_pdb(cyclobutane(SITE, diagonal=1.95)))
    assert ligand.GetNumBonds() == 5  # the guessed C1-C3 bond
    fixed = assign_bond_orders_from_template(Chem.MolFromSmiles("C1CCC1"), ligand)
    assert fixed.GetNumBonds() == 4
    assert _canonical(fixed) == "C1CCC1"


def test_aromatic_bond_orders_assigned():
    ligand = _pdb_ligand(complex_pdb(benzene(SITE)))
    fixed = assign_bond_orders_from_template(Chem.MolFromSmiles("c1ccccc1"), ligand)
    assert all(b.GetIsAromatic() for b in fixed.GetBonds())
    assert fixed.GetConformer().GetPositions() == pytest.approx(
        ligand.GetConformer().GetPositions()
    )


@pytest.mark.parametrize("with_hydrogens", [False, True])
@pytest.mark.parametrize(
    "smiles", ["c1cc[nH]c1", "CC(=O)Nc1ccc(O)cc1", "C[N+](C)(C)CC(=O)[O-]"]
)
def test_bond_orders_with_and_without_explicit_hydrogens(smiles, with_hydrogens):
    ligand = _embedded_pdb_ligand(smiles, with_hydrogens)
    fixed = assign_bond_orders_from_template(Chem.MolFromSmiles(smiles), ligand)
    assert _canonical(fixed) == Chem.CanonSmiles(smiles)
    # Explicit hydrogens and coordinates are kept.
    assert fixed.GetNumAtoms() == ligand.GetNumAtoms()


@pytest.mark.parametrize(
    "wrong",
    [
        "c1ccncc1",  # different element
        "C1CCCCC1C",  # different atom count
        "CCCCCC",  # would only fit by deleting a real 1.39 Å ring bond
    ],
)
def test_mismatched_smiles_raises(wrong):
    ligand = _pdb_ligand(complex_pdb(benzene(SITE)))
    with pytest.raises(ValueError):
        assign_bond_orders_from_template(Chem.MolFromSmiles(wrong), ligand)


def test_ligands_from_pdb_one_per_residue(tmp_path):
    path = tmp_path / "ref.pdb"
    path.write_text(
        complex_pdb(benzene(SITE, "B", 1), benzene(SITE_2, "C", 1, first_serial=600))
    )
    ligands = ligands_from_pdb(str(path), "c1ccccc1")
    assert [lig.GetNumAtoms() for lig in ligands] == [6, 6]
    centres = [lig.GetConformer().GetPositions().mean(axis=0) for lig in ligands]
    assert centres[0] == pytest.approx(SITE, abs=1e-3)
    assert centres[1] == pytest.approx(SITE_2, abs=1e-3)


@pytest.mark.parametrize(
    ("pdb_text", "smiles"),
    [
        (complex_pdb(benzene(SITE)), "not a smiles"),
        (complex_pdb(benzene(SITE)), "c1ccncc1"),
        (complex_pdb(), "c1ccccc1"),  # no LIG residue
    ],
)
def test_ligands_from_pdb_returns_none_when_it_cant(tmp_path, pdb_text, smiles):
    path = tmp_path / "x.pdb"
    path.write_text(pdb_text)
    assert ligands_from_pdb(str(path), smiles) is None
