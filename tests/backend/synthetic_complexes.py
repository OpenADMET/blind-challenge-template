"""Fully synthetic protein-ligand PDB text for structure-track tests.

A short poly-alanine strand plus benzene rings named ``LIG``: enough for
OpenStructure to load, find a binding site and score, with no real structural data.
"""

import math

# Ligand centres near the strand. SITE and SITE_2 are two separate binding sites;
# OFFSET is SITE shifted by 3 Å, a plausible but wrong pose.
SITE = (9.5, 4.0, 3.0)
OFFSET = (12.5, 4.0, 3.0)
SITE_2 = (13.0, -3.5, -1.5)


def _peptide_lines() -> list[str]:
    lines = []
    serial = 1
    for res in range(6):
        x0 = res * 3.8
        for name, element, dx, dy, dz in [
            ("N", "N", 0.0, 0.0, 0.0),
            ("CA", "C", 1.46, 0.0, 0.0),
            ("C", "C", 2.0, 1.42, 0.0),
            ("O", "O", 1.25, 2.39, 0.0),
            ("CB", "C", 1.9, -0.8, 1.2),
        ]:
            lines.append(
                f"ATOM  {serial:5d}  {name:<3} ALA A{res + 1:4d}    "
                f"{x0 + dx:8.3f}{dy:8.3f}{dz:8.3f}  1.00  0.00           {element:>2}"
            )
            serial += 1
    return lines


def benzene(
    centre: tuple[float, float, float],
    chain: str = "B",
    resnum: int = 1,
    altloc: str = " ",
    first_serial: int = 500,
) -> list[str]:
    """HETATM lines for one benzene ring named LIG, centred on ``centre``."""
    cx, cy, cz = centre
    lines = []
    for i in range(6):
        angle = 2 * math.pi * i / 6
        x, y = cx + 1.39 * math.cos(angle), cy + 1.39 * math.sin(angle)
        lines.append(
            f"HETATM{first_serial + i:5d}  C{i + 1:<2}{altloc}LIG {chain}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{cz:8.3f}  1.00  0.00           C "
        )
    return lines


def complex_pdb(*ligands: list[str]) -> str:
    """A PDB file with the peptide and the given ligand residues."""
    lines = _peptide_lines() + [line for ligand in ligands for line in ligand]
    return "\n".join([*lines, "END"]) + "\n"


def cyclobutane(
    centre: tuple[float, float, float],
    diagonal: float,
    chain: str = "B",
    resnum: int = 1,
    first_serial: int = 500,
) -> list[str]:
    """HETATM lines for a flattened four-carbon ring named LIG (SMILES ``C1CCC1``).

    The ring is a rhombus with 1.55 Å sides. ``diagonal`` sets the C1-C3 distance;
    below about 2.08 Å OpenStructure's distance-based bonding adds a false C1-C3
    bond, as it can across a real strained ring. Keep the other diagonal (C2-C4)
    above that too: for ``diagonal`` between about 2.1 and 2.25 Å neither has a
    false bond.
    """
    side = 1.55
    other = 2 * math.sqrt(side**2 - (diagonal / 2) ** 2)
    cx, cy, cz = centre
    corners = [
        (cx + diagonal / 2, cy, cz),
        (cx, cy + other / 2, cz),
        (cx - diagonal / 2, cy, cz),
        (cx, cy - other / 2, cz),
    ]
    return [
        f"HETATM{first_serial + i:5d}  C{i + 1:<2} LIG {chain}{resnum:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C "
        for i, (x, y, z) in enumerate(corners)
    ]
