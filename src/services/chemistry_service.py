from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def is_valid_smiles(s: str) -> bool:
    if not s or s.strip().isdigit():
        return False
    return Chem.MolFromSmiles(s) is not None


def get_svg(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(440, 280)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1.8
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def compute_properties(smiles: str) -> dict | None:
    if not is_valid_smiles(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return {
        "Mol. Weight":      round(Descriptors.MolWt(mol), 2),
        "LogP":             round(Descriptors.MolLogP(mol), 2),
        "H-Bond Donors":    Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "TPSA (Å²)":        round(rdMolDescriptors.CalcTPSA(mol), 2),
        "Rotatable Bonds":  Lipinski.NumRotatableBonds(mol),
    }


def lipinski_pass(props: dict) -> bool:
    if not props:
        return False
    return sum([
        props["Mol. Weight"]      <= 500,
        props["LogP"]             <= 5,
        props["H-Bond Donors"]    <= 5,
        props["H-Bond Acceptors"] <= 10,
    ]) >= 4