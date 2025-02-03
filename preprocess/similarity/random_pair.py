import pandas as pd
import random

import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdFMCS

def generate_smiles_pairs(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    smiles_list = data['smiles'].tolist()
    selected_smiles = random.sample(smiles_list, 40000)
    pairs = [(random.choice(selected_smiles), random.choice(selected_smiles)) for _ in range(20000)]
    pairs_df = pd.DataFrame(pairs, columns=['smiles_1', 'smiles_2'])

    def compute_tanimoto_distance(smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None
        fp1 = GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return 1 - similarity

    def compute_mcs_atoms(smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None
        mcs_result = rdFMCS.FindMCS([mol1, mol2], completeRingsOnly=True, matchValences=True)
        return mcs_result.numAtoms if mcs_result is not None else None

    def compute_dice_similarity(smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None
        fp1 = GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        return DataStructs.DiceSimilarity(fp1, fp2)

    def compute_cosine_similarity(smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None
        fp1 = GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        numerator = sum(a & b for a, b in zip(fp1, fp2))
        denominator = (sum(fp1) ** 0.5) * (sum(fp2) ** 0.5)
        return numerator / denominator if denominator != 0 else None

    pairs_df['Tanimoto_distance'] = pairs_df.apply(
        lambda row: compute_tanimoto_distance(row['smiles_1'], row['smiles_2']), axis=1
    )

    pairs_df['MCS_num_atoms'] = pairs_df.apply(
        lambda row: compute_mcs_atoms(row['smiles_1'], row['smiles_2']), axis=1
    )

    pairs_df['Dice_similarity'] = pairs_df.apply(
        lambda row: compute_dice_similarity(row['smiles_1'], row['smiles_2']), axis=1
    )

    pairs_df['Cosine_similarity'] = pairs_df.apply(
        lambda row: compute_cosine_similarity(row['smiles_1'], row['smiles_2']), axis=1
    )
    pairs_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = "xxx/dataset/ZINC/ZINC_250k.csv"
    output_csv = "xxx/model/pretrain/evaluation/zinc-20k.csv"
    generate_smiles_pairs(input_csv=input_csv, output_csv=output_csv)