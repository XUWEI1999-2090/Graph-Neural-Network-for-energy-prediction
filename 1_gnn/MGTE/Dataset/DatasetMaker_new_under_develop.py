# coding=utf-8
from collections import defaultdict
import numpy as np
import pickle
from rdkit.Chem import AllChem as Chem


class RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetMaker(object):
    '''
    INFO:一种fingerprint就是一个词！一个分子就是一句话，fingerprint编码就是在分词，
     从图模型进行分词！按照不同的radius！
    ref: https://github.com/masashitsubaki/GNN_molecules

    '''

    def transform_mol_graph(self, mol_list):
        '''
        这里的transform 和fit transform的区别就是，这里使用的不再是default dict
        而是普通的dict，因为需要保证进来的mol不会出现新的fingerprint
        :return:
        '''
        self.fingerprint_dict = dict(self.fingerprint_dict)
        self.bond_dict = dict(self.bond_dict)
        self.atom_dict = dict(self.atom_dict)
        self.edge_dict = dict(self.edge_dict)
        return self.fit_transform_mol_graph(mol_list, self.fingerprint_radius)

    @staticmethod
    def fit_transform_property(y):
        all_properties_data = []
        for property in y:
            all_properties_data.append([[float(property)]])
        all_properties_data = np.array(all_properties_data)
        mean, std = np.mean(all_properties_data), np.std(all_properties_data)
        all_properties_data = (all_properties_data - mean) / std
        return all_properties_data, mean, std

    def fit_transform_mol_graph(self, mol_list, fingerprint_radius=1, debug=False):
        self.fingerprint_radius = fingerprint_radius
        for i in range(len(mol_list)):
            mol = mol_list[i]
            atoms = self.__create_atoms(mol)
            ij_bond_dict = self.__create_ij_bond_dict(mol)
            fingerprint = self.__extract_fingerprints(
                atoms, ij_bond_dict, self.fingerprint_radius
            )
            self.all_fingerprint_data.append(fingerprint)
            adjacency = self.__create_adjacency(mol)
            self.all_adjacencies_data.append(adjacency)
            if debug:
                print("Fingerprint")
                print(fingerprint)
                print("ij bond dict")
                print(ij_bond_dict)
                print("Adjacency")
                print(adjacency)
        return self.all_fingerprint_data, self.all_adjacencies_data

    def __init__(self):
        '''

        :param mol_string_list: 这里的mol是RDKit的Chem.Mol
        :param property_list: 浮点数的list，每个对应于上面的每个mol
        '''
        # TODO： 把default dict转成普通的dict！！！
        # 这些dict实际上就是当出现一个独特的key时，将value设为len(atom_dict)，实际上就是随意添加key，value是0,1,2,3, ...这样
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        self.all_fingerprint_data = []
        self.all_adjacencies_data = []
        self.all_properties_data = []

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def save_data_as_npy_and_pkl_files(self, save_dir):
        pass


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

    def __create_atoms(self, mol):
        """Create a list of atom (e.g., hydrogen and oxygen) IDs"""
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        # 这里如果dict中有key，就会给出相应的index，否则创建一个新的，给出len作为value
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)

    def __create_ij_bond_dict(self, mol):
        """Create a dictionary, which each key is a node ID
            and each value is the tuples of its neighboring node
            and bond (e.g., single and double) IDs."""
        ij_bond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            ij_bond_dict[i].append((j, bond))
            ij_bond_dict[j].append((i, bond))
        return ij_bond_dict

    def __extract_fingerprints(self, atoms, ij_bond_dict, radius):
        """
        相当于图模型分词分成不同的fingerprint，然后去embedding，将fingerprint之间的关系拿出来
        然后再去得到分子embedding的特征
        Extract the r-radius subgraphs (i.e., fingerprints)
            from a molecular graph using Weisfeiler-Lehman algorithm."""
        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [self.fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            ij_edge_dict = ij_bond_dict

            for _ in range(radius):

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

    def __create_adjacency(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)
