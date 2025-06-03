from collections import defaultdict
import numpy as np
import pickle
from rdkit.Chem import AllChem as Chem
import os

class RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetMaker(object):
    '''
    INFO:一种fingerprint就是一个词！一个分子就是一句话，fingerprint编码就是在分词，
     从图模型进行分词！按照不同的radius！
    ref: https://github.com/masashitsubaki/GNN_molecules

    '''

    def fit(self):
        # TODO:拟合得到所有atom dict，edge dict等等
        pass

    def transform(self):
        # TODO： 将拟合得到的dict用于编码图模型得到矩阵
        pass

    def __init__(self, mol_list, property_list, metal_label, mol_name_list=None, fingerprint_radius=1, debug=False):
        '''

        :param mol_list: 这里的mol是RDKit的Chem.Mol
        :param property_list: 浮点数的list，每个对应于上面的每个mol
        '''
        assert len(mol_list) == len(property_list)
        if mol_name_list is not None:
            assert len(mol_name_list) == len(mol_list)
        self.mol_name_list = mol_name_list
        self.metal_label = metal_label
        self.mol_list = mol_list
        self.property_list = property_list
        self.radius = fingerprint_radius
        # TODO： 把default dict转成普通的dict！！！
        # 这些dict实际上就是当出现一个独特的key时，将value设为len(atom_dict)，实际上就是随意添加key，value是0,1,2,3, ...这样
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        self.all_fingerprint_data = []
        self.all_adjacencies_data = []
        self.all_properties_data = []

        for i in range(len(self.mol_list)):
            mol = self.mol_list[i]
            property = self.property_list[i]
            atoms = self.create_atoms(mol)
            ij_bond_dict = self.create_ij_bond_dict(mol)
            fingerprint = self.extract_fingerprints(
                atoms, ij_bond_dict
            )

            self.all_fingerprint_data.append(fingerprint)
            adjacency = self.create_adjacency(mol)
            self.all_adjacencies_data.append(adjacency)
            self.all_properties_data.append([[float(property)]])

            if debug:
                print("Fingerprint")
                print(fingerprint)
                print("ij bond dict")
                print(ij_bond_dict)
                print("Adjacency")
                print(adjacency)
        self.all_properties_data = np.array(self.all_properties_data)
        self.mean, self.std = np.mean(self.all_properties_data), np.std(self.all_properties_data)
        self.all_properties_data = (self.all_properties_data - self.mean) / self.std

    def save_data_as_npy_and_pkl_files(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.save(save_dir + '/molecules', self.all_fingerprint_data)
        np.save(save_dir + '/adjacencies', self.all_adjacencies_data)
        np.save(save_dir + '/properties', self.all_properties_data)
        np.save(save_dir + '/mean', self.mean)
        np.save(save_dir + '/std', self.std)
        with open(save_dir + "/fingerprint_dict.pkl", "wb") as f:
            pickle.dump(dict(self.fingerprint_dict), f)
        with open(save_dir + "./mol_name_list.pkl", "wb") as f:
            pickle.dump(self.mol_name_list, f)
        with open(save_dir + "./metal_label.pkl", "wb") as f:
            pickle.dump(self.metal_label, f)

        # data = {}
        # data["fingerprints"] = self.all_fingerprint_data
        # data["adjacency"] = self.all_adjacencies_data
        # data["y"] = self.all_properties_data
        # data["y_mean"] = self.mean
        # data["y_std"] = self.std
        # data["fingerprint_dict"] = dict(self.fingerprint_dict)
        # data["edge_dict"] = dict(self.edge_dict)
        # data["fingerprint_radius"] = self.radius
        # with open(filename, "wb") as f:
        #     pickle.dump(data, f)

    def create_atoms(self, mol):
        """Create a list of atom (e.g., hydrogen and oxygen) IDs"""
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        # 这里如果dict中有key，就会给出相应的index，否则创建一个新的，给出len作为value
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)

    def create_ij_bond_dict(self, mol):
        """Create a dictionary, which each key is a node ID
            and each value is the tuples of its neighboring node
            and bond (e.g., single and double) IDs."""
        ij_bond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            ij_bond_dict[i].append((j, bond))
            ij_bond_dict[j].append((i, bond))
        return ij_bond_dict # 为什么是ij_bond_dict而不是bond

    def extract_fingerprints(self, atoms, ij_bond_dict):
        """
        相当于图模型分词分成不同的fingerprint，然后去embedding，将fingerprint之间的关系拿出来
        然后再去得到分子embedding的特征
        Extract the r-radius subgraphs (i.e., fingerprints)
            from a molecular graph using Weisfeiler-Lehman algorithm."""
        if (len(atoms) == 1) or (self.radius == 0):
            fingerprints = [self.fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            ij_edge_dict = ij_bond_dict

            for _ in range(self.radius):

                """Update each node ID considering its neighboring nodes and edges
                (i.e., r-radius subgraphs or fingerprints)."""
                fingerprints = []
                for i, j_edge in ij_edge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprint_dict[fingerprint])
                nodes = fingerprints

                """Also update each edge ID considering two nodes
                on its both sides."""
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in ij_edge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = self.edge_dict[(both_side, edge)]
                        _i_jedge_dict[i].append((j, edge))
                ij_edge_dict = _i_jedge_dict

        return np.array(fingerprints)

    def create_adjacency(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)
