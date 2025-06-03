        metal_feature = torch.tensor(np.transpose([metal_feature]))
        #print(metal_feature) #是batch_size所以每次36个数
        #print(self.f_molecular_vectors.shape, metal_feature.shape)
        #new_feature= np.ones([len(metal_feature),17], dtype = float)
        new_feature = []
        for i in range(len(metal_feature)):
            a = self.f_molecular_vectors.detach()[i]
            b = metal_feature[i]
            new_feature.append(np.concatenate((a,b)))
        new_feature = torch.tensor(new_feature, dtype=torch.float32)
        #print(new_feature.shape)
        molecular_properties = self.W_property(new_feature)
        #print(len(molecular_properties))