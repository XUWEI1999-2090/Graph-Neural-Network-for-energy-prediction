import pickle

from MEACRNG.Tools.MolViz import show_mol_array_using_PIL, show_mol
with open("E:/ethnaol reforming/C2_2.1/mol_id_string_on_testset_1.17.pkl","rb") as f:
    a = pickle.load(f)

with open("E:/ethnaol reforming/C2_3.3/molecule_dataset.pkl", "rb") as f:
    metal, mol_list, mol_string, energy = pickle.load(f)
print(len(mol_list),len(mol_string),len(energy))
# 随机选择进行visualize
# target_index = random.choices(population=range(len(mol_list)),k=36)
#
# show_mol_array_using_PIL([mol_list[i] for i in target_index],
#                species_name_replace=[mol_string[i] for i in target_index])

# 按照顺序来
# 不要第一个C
mol_list = mol_list[1:]
mol_string = mol_string[1:]
# 30+ 45*7 + 15
k = 45
n = (len(mol_list)-30) // k

# for i in range(n):
#     print("From %s to %s"%(30+ k * i,30+ (i + 1) * k), i)
#     show_mol_array_using_PIL(mol_list[30+ k * i:30+ (i + 1) * k],
#                              species_name_replace=mol_string[30+ k * i: 30+ (i + 1) * k],
#                              save_filename="E://2023毕业论文/305/figs_%s.png" % i)
#


# print("From %s to %s"%(k * n,len(mol_list)))
# show_mol_array_using_PIL(mol_list[k * n:],
#                          species_name_replace=mol_string[k * n:],
#                          save_filename="figs_final.png")
print("From 345 to 360")
show_mol_array_using_PIL(mol_list[345:],
                         species_name_replace=mol_string[345:],
                         save_filename="E://2023毕业论文/305/figs_final.png")
# from rdkit.Chem import Draw
#
# for i in range(len(mol_string)):
#     if mol_string[i] == 'CH2C(-H)OH[Pt]':
#         mol = mol_list[i]
#         Draw.DrawingOptions.atomLabelFontSize = 20
#         Draw.DrawingOptions.bondLineWidth = 2
#         Draw.DrawingOptions.dash = (8, 8)
#         img = Draw.MolToImage(mol, size=(600, 600), fitImage=True)
#         img.show()