"""Ligand bonds from the ground-truth SMILES, for OST and PoseBusters.

Every ligand in a submission is a residue named ``LIG``, which isn't in
OpenStructure's compound library, and PDB files from structure-prediction tools
have no ``CONECT`` records. So both OST and RDKit guess the ligand's bonds from
interatomic distances. That guess goes wrong for strained rings: in an azetidine,
cyclobutane or some five-membered aromatics, the two atoms across the ring can sit
closer than OST's bonding cutoff (about 2.08 Å for carbon), and a bond that doesn't
exist is added. When it is added in one structure but not the other (the reference
and the prediction are different conformations), the two ligand graphs no longer
match, both OST scorers find no assignment, and the compound scores as
unscoreable. A defect in the reference makes the compound unscoreable for every
entrant.

This module takes the bonds from the SMILES instead: the ligand atoms are read
from the PDB, matched to the SMILES, given the SMILES' bonds and bond orders, and
handed to OST with explicit connectivity. The same bond orders are used for
PoseBusters, whose geometry and energy checks need them.

``assign_bond_orders_from_template`` is adapted from the CYP challenge
(``cyp-challenge@9a5b8de``), which used it for PoseBusters. The approach of
rebuilding both ligands from the SMILES before OST scoring follows the OpenBind
challenge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from rdkit import Chem  # type: ignore[import]

LIGAND_RESNAME = "LIG"

# A bond in the PDB ligand that isn't in the SMILES is treated as guessed across a
# strained ring, and removed, only if it is longer than this many times the sum
# of the two atoms' covalent radii (1.15 x 1.52 = 1.75 Å for C-C). Real bonds are
# shorter; the false bonds OpenStructure and RDKit add are about 1.9-2.1 Å.
FALSE_BOND_FACTOR = 1.15


def assign_bond_orders_from_template(refmol: Chem.Mol, mol: Chem.Mol) -> Chem.Mol:
    """Give ``mol`` exactly the bonds and bond orders of ``refmol``.

    Like ``AllChem.AssignBondOrdersFromTemplate``, with three differences that
    matter for ligands read from PDB files:

    * Every topological match is tried, not just the first. A locally symmetric
      ring (an NH-pyrrole read either way round) can match in several ways, and
      only some give a valid Kekulé structure. The first match that sanitizes is
      kept.
    * Matching happens on heavy atoms. A PDB ligand may carry explicit hydrogens
      that the SMILES has as implicit counts; copying the SMILES' hydrogen counts
      onto atoms that already have hydrogen neighbours double-counts them and
      raises valence errors.
    * A bond in ``mol`` that isn't in ``refmol`` is removed if it is stretched
      (longer than ``FALSE_BOND_FACTOR`` times the sum of the two atoms' covalent
      radii, about 1.75 Å for C-C). That is a bond guessed from distances across a
      strained ring, about 1.9-2.1 Å long. It otherwise survives, because a
      substructure match doesn't object to ``mol`` having extra bonds. An extra
      bond of normal length is real, so the ligand is not the SMILES compound and
      the match is rejected; otherwise a SMILES with fewer bonds (hexane) could
      be forced onto a ligand that has more (benzene).

    No bond is ever added: a substructure match only succeeds when every bond of
    ``refmol`` is already in ``mol``.

    Unlike the CYP original, there is no shortcut for a ligand that already
    matches with its bond orders intact: for a saturated strained ring the PDB
    ligand matches the SMILES even with the false bond, and the shortcut kept it.

    Parameters
    ----------
    refmol : Chem.Mol
        The template, with correct bonds (from SMILES).
    mol : Chem.Mol
        The ligand read from a PDB file, with guessed bonds; may include explicit
        hydrogens.

    Returns
    -------
    Chem.Mol
        A sanitized copy of ``mol`` (hydrogens and 3D coordinates kept) with
        bonds, bond orders, aromaticity and formal charges from ``refmol``.

    Raises
    ------
    ValueError
        If the heavy atoms of ``mol`` don't match ``refmol`` topologically, every
        match would remove a bond of normal length, or no match gives a molecule
        that sanitizes.

    """
    from rdkit import Chem

    periodic_table = Chem.GetPeriodicTable()
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetIntProp("_origIdx", atom.GetIdx())
    mol_noh = Chem.RemoveHs(mol, sanitize=False)
    heavy_to_orig = [atom.GetIntProp("_origIdx") for atom in mol_noh.GetAtoms()]

    if mol_noh.GetNumAtoms() != refmol.GetNumAtoms():
        raise ValueError(
            f"Ligand has {mol_noh.GetNumAtoms()} heavy atoms but the SMILES has "
            f"{refmol.GetNumAtoms()}"
        )

    # Flatten both to single, non-aromatic, neutral bonds and atoms to find
    # topological (bond-order-agnostic) matches.
    flat_ref, flat_mol = Chem.Mol(refmol), Chem.Mol(mol_noh)
    for flat in (flat_ref, flat_mol):
        for bond in flat.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        for atom in flat.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
    matches = flat_mol.GetSubstructMatches(flat_ref, uniquify=False, maxMatches=1000)
    if not matches:
        raise ValueError("The ligand's atoms don't match the SMILES")

    assigned = None
    errors: list[str] = []
    for match in matches:
        candidate = Chem.RWMol(mol_noh)
        ref_edges = {
            frozenset((match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]))
            for b in refmol.GetBonds()
        }
        # Every heavy atom is matched (the counts are equal), so a bond not in the
        # template is either guessed across a strained ring (stretched: remove it)
        # or real (this match, and maybe the SMILES, is wrong).
        positions = mol_noh.GetConformer().GetPositions()
        real_extra_bond = None
        for bond in list(candidate.GetBonds()):
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if frozenset((a1, a2)) in ref_edges:
                continue
            length = float(((positions[a1] - positions[a2]) ** 2).sum() ** 0.5)
            normal = sum(
                periodic_table.GetRcovalent(candidate.GetAtomWithIdx(i).GetAtomicNum())
                for i in (a1, a2)
            )
            if length <= FALSE_BOND_FACTOR * normal:
                real_extra_bond = (a1, a2, length)
                break
            candidate.RemoveBond(a1, a2)
        if real_extra_bond is not None:
            errors.append(
                "bond between atoms {} and {} ({:.2f} Å) is not in the SMILES".format(
                    *real_extra_bond
                )
            )
            continue
        for ref_bond in refmol.GetBonds():
            a1, a2 = match[ref_bond.GetBeginAtomIdx()], match[ref_bond.GetEndAtomIdx()]
            if candidate.GetBondBetweenAtoms(a1, a2) is None:
                candidate.AddBond(a1, a2, ref_bond.GetBondType())
            bond = candidate.GetBondBetweenAtoms(a1, a2)
            bond.SetBondType(ref_bond.GetBondType())
            bond.SetIsAromatic(ref_bond.GetIsAromatic())
        for ref_atom in refmol.GetAtoms():
            atom = candidate.GetAtomWithIdx(match[ref_atom.GetIdx()])
            atom.SetIsAromatic(ref_atom.GetIsAromatic())
            atom.SetFormalCharge(ref_atom.GetFormalCharge())
            atom.SetNumExplicitHs(ref_atom.GetNumExplicitHs())
            atom.SetNoImplicit(False)
        try:
            heavy = candidate.GetMol()
            Chem.SanitizeMol(heavy)
        except Exception as exc:  # noqa: BLE001 - try the next match
            errors.append(str(exc))
            continue
        assigned = heavy
        break

    if assigned is None:
        raise ValueError(
            f"None of the {len(matches)} matches to the SMILES sanitized: "
            + "; ".join(sorted(set(errors)))
        )

    if mol.GetNumAtoms() == assigned.GetNumAtoms():
        return assigned

    # Copy the heavy-atom bonds back onto the molecule with its explicit
    # hydrogens, mapping atoms by original index (RemoveHs renumbers them).
    result = Chem.RWMol(mol)
    heavy_orig = set(heavy_to_orig)
    heavy_bonds = {
        frozenset(
            (heavy_to_orig[b.GetBeginAtomIdx()], heavy_to_orig[b.GetEndAtomIdx()])
        )
        for b in assigned.GetBonds()
    }
    for bond in list(result.GetBonds()):
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (
            a1 in heavy_orig
            and a2 in heavy_orig
            and frozenset((a1, a2)) not in heavy_bonds
        ):
            result.RemoveBond(a1, a2)
    for heavy_bond in assigned.GetBonds():
        a1 = heavy_to_orig[heavy_bond.GetBeginAtomIdx()]
        a2 = heavy_to_orig[heavy_bond.GetEndAtomIdx()]
        if result.GetBondBetweenAtoms(a1, a2) is None:
            result.AddBond(a1, a2, heavy_bond.GetBondType())
        bond = result.GetBondBetweenAtoms(a1, a2)
        bond.SetBondType(heavy_bond.GetBondType())
        bond.SetIsAromatic(heavy_bond.GetIsAromatic())
    for heavy_atom in assigned.GetAtoms():
        atom = result.GetAtomWithIdx(heavy_to_orig[heavy_atom.GetIdx()])
        atom.SetIsAromatic(heavy_atom.GetIsAromatic())
        atom.SetFormalCharge(heavy_atom.GetFormalCharge())
        # The explicit hydrogen atoms are real neighbours now, so don't also keep
        # the SMILES' hydrogen counts (that would double-count them).
        atom.SetNumExplicitHs(0)
    final = result.GetMol()
    Chem.SanitizeMol(final)
    return final


def ligands_from_pdb(pdb_path: str, smiles: str) -> list[Chem.Mol] | None:
    """Read each ``LIG`` residue in a PDB file, with bonds from ``smiles``.

    A reference structure can hold several copies of the ligand (e.g. symmetry
    mates), so each ``LIG`` residue (chain, residue number and insertion code) is
    returned separately. Only the first alternate location of each atom is read.

    Parameters
    ----------
    pdb_path : str
        Path to a complex PDB file.
    smiles : str
        Ground-truth SMILES for the ligand.

    Returns
    -------
    list[Chem.Mol] | None
        One molecule per ``LIG`` residue, in file order, with 3D coordinates,
        explicit hydrogens if the file has them, and bonds from ``smiles``. None
        if the file can't be read, has no ``LIG`` residue, the SMILES can't be
        parsed, or any copy doesn't match the SMILES; the reason is logged.

    """
    from rdkit import Chem

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        logger.warning("Could not parse SMILES {!r} for {}", smiles, pdb_path)
        return None
    complex_mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if complex_mol is None:
        logger.warning("RDKit could not read {}", pdb_path)
        return None

    residues: dict[tuple[str, int, str], list[int]] = {}
    for atom in complex_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None and info.GetResidueName().strip() == LIGAND_RESNAME:
            key = (info.GetChainId(), info.GetResidueNumber(), info.GetInsertionCode())
            residues.setdefault(key, []).append(atom.GetIdx())
    if not residues:
        logger.warning("No {} residue in {}", LIGAND_RESNAME, pdb_path)
        return None

    ligands = []
    for key, atom_indices in residues.items():
        keep = set(atom_indices)
        ligand = Chem.RWMol(complex_mol)
        for idx in sorted(range(ligand.GetNumAtoms()), reverse=True):
            if idx not in keep:
                ligand.RemoveAtom(idx)
        try:
            ligands.append(assign_bond_orders_from_template(template, ligand.GetMol()))
        except Exception as exc:  # noqa: BLE001 - caller falls back on None
            logger.warning(
                "Could not give {} residue {} in {} the bonds of {!r}: {}",
                LIGAND_RESNAME,
                key,
                pdb_path,
                smiles,
                exc,
            )
            return None
    return ligands


def ost_ligand(mol: Chem.Mol, chain_name: str) -> Any:
    """Convert an RDKit ligand to an OST entity with the same bonds.

    Goes through an SDF block, which OST reads connectivity from rather than
    guessing it. Hydrogens are dropped, as in the ``LIG`` selection used without
    SMILES, because protonation differs between prediction tools and crystal
    structures.

    Parameters
    ----------
    mol : Chem.Mol
        A ligand from ``ligands_from_pdb``.
    chain_name : str
        Name for the entity's single chain, unique among the ligands passed to
        one scorer. Entities read from SDF all get the same chain name and
        residue number, and ``score_single_structure`` pairs the two scorers'
        results by chain name and residue number.

    Returns
    -------
    Any
        An ``ost.mol.EntityHandle``: the ligand's heavy atoms as a standalone
        entity, with its residue named ``LIG``.

    """
    from ost import io, mol as ost_mol  # type: ignore[import]
    from rdkit import Chem

    entity = io.SDFStrToEntity(Chem.MolToMolBlock(Chem.RemoveHs(mol)))
    entity = ost_mol.CreateEntityFromView(entity.Select("ele!=H"), False)
    editor = entity.EditXCS()
    editor.RenameChain(entity.chains[0], chain_name)
    editor.RenameResidue(entity.residues[0], LIGAND_RESNAME)
    editor.UpdateICS()
    return entity
