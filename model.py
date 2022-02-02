import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_size, ncf=False):
        super(GMF, self).__init__()
        self.ncf = ncf
        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_size)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_size)

        self.linear = nn.Linear(in_features=embedding_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)
        element_product = torch.mul(user_embedding, item_embedding) 
        output = self.linear(element_product)
        if ncf == True:
            return output
        return self.sigmoid(output)


class MLP(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_layers, dropout=0, ncf=False):
        """
            n_users: number of user
            n_items: number of item
            embedding_dim: embedding size
            n_layers: number of layer
            dropout: dropout rate
            ncf: boolean, NCF or not
        """
        super(MLP, self).__init__()
        self.ncf = ncf
        self.user_embedding = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim
            )
        self.item_embedding = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim
            )
        self.fc_layers = nn.ModuleList()
        
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            if dropout != 0:
                self.fc_layers.append(nn.Dropout(p=self.dropout))
            self.fc_layers.append(nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.linear = nn.Linear(in_features=layers[-1], out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        user_embedding = self.user_embedding(user_input)
        item_embedding = self.item_embedding(item_input)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for i in range(len(self.fc_layers)):
            vector = self.fc_layers[i](vector)
            vector = self.relu(vector)
        output = self.linear(vector)
        if self.ncf == True:
            return output
        return self.sigmoid(output)


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, embedding_size, mlp_layers=[256, 128, 64]):
        super(NeuMF, self).__init__()
        self.gmf = GMF(n_users, n_items, embedding_size, ncf=True)
        self.mlp = MLP(n_users, n_items, embedding_size, mlp_layers, ncf=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        gmf = self.gmf(user_input, item_input)
        mlp = self.mlp(user_input, item_input)
        concat = torch.cat([gmf, mlp], dim=-1)
        linear = nn.Linear(in_features=concat.shape[1], out_features=1)
        output = linear(concat)
        output = self.sigmoid(output)
        return output.squeeze()



