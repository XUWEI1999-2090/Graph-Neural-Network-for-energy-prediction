# INFO 1. 载入文本得到分子的string以及相应的能量，
'''
with open("./set_corrected/new_C1.csv", "r", encoding="utf-8") as f:
    for i in f.readlines():
        try:
            data = i.split(",")
            mol_string = data[1]
            metal_label = data[0]
            Head_Carbon = mol_string.find('C')
            l = list(mol_string)
            if '-' in l:
                if '(' in l:
                    l = l[:Head_Carbon + 1] + ['(', '-', metal_label, ')'] + l[Head_Carbon + 1:]
                else:
                    l = l + ['(', '-', metal_label, ')']
            else:
                l = l[:Head_Carbon + 1] + ['(', '-', metal_label, ')'] + l[Head_Carbon + 1:]
            mol_string = "".join(l)
            print(mol_string)
        except:
            pass

mol_string_for_test = ['CH3CH2O-H','CH3CH(-H)OH','H-CH2CH2OH','CH3CH2O','CH3CHOH','CH2CH2OH',
                       'CH3CH(-H)O','H-CH2CHOH','CH(-H)CH2OH','CH3CHO','CH2CHOH','CHCH2OH',
                       'CH3C(-H)O','CH2CHO-H','H-CCH2OH','CH3CO','CH2CHO','CCH2OH',
                       'H-CH2CO','CH2C(-H)O','CCH(-H)OH','CH2CO','CCHO-H','CHCO','CCHO',
                       'H-CCO','CC(-H)O','CCO','C-CO','CH-CO']
metal_label = 'K'
mol = []
for i in mol_string_for_test:
    Head_Carbon = i.find('C')
    l = list(i)
    if '-' in l:
        if '(' in l:
            l = l[:Head_Carbon + 1] + ['(', '-', metal_label, ')'] + l[Head_Carbon + 1:]
        else:
            l = l + ['(', '-', metal_label, ')']
    else:
        l = l[:Head_Carbon + 1] + ['(', '-', metal_label, ')'] + l[Head_Carbon + 1:]
    mol_string = "".join(l)
    mol.append(mol_string)
count=0 #设置初始计数
for j in mol:
    print("'%s'"%j, end=',')
    count += 1 #开始计数
    if count % 6 == 0: #每10个换行
        print(end='\n')

'''
import csv

f = csv.reader(open("E:/ethnaol reforming/C2_3.0/C2_3.0.csv",'r'))
wf = csv.writer(open("E:/ethnaol reforming/C2_3.0/C2_3.2.csv",'w',newline=''))

for i in f:
    mol_string = i[1]
    metal_label = i[0]
    Head_Carbon = mol_string.find('C')
    l = list(mol_string)
    if '-' in l:
        if '(' in l:
            l = l[:Head_Carbon + 1] + ['(', '-', metal_label, ')'] + l[Head_Carbon + 1:]
        else:
            l = l + ['(', '-', metal_label, ')']
    else:
        l = l[:Head_Carbon + 1] + ['(', '-', metal_label, ')'] + l[Head_Carbon + 1:]
    mol_string = "".join(l)
    print(mol_string)
    wf.writerow([metal_label,mol_string,i[2]])

