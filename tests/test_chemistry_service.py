from src.services.chemistry_service import (
    is_valid_smiles,
    lipinski_pass,
    compute_properties,
)


def test_valid_smiles():
    assert is_valid_smiles("CCO")


def test_invalid_smiles():
    assert not is_valid_smiles("not-a-smiles")


def test_properties_are_computed():
    props = compute_properties("CCO")
    assert props is not None
    assert props["Mol. Weight"] > 0


def test_lipinski_for_ethanol():
    props = compute_properties("CCO")
    assert lipinski_pass(props)