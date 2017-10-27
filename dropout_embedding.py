from allennlp.modules.token_embedders import TokenEmbedder
from overrides import overrides
from torch.nn import Dropout

class DropoutEmbedding(TokenEmbedder):
    def __init__(self, base: TokenEmbedder, dropout_rate: float):
        super(DropoutEmbedding, self).__init__()
        self.base = base
        self.dropout = Dropout(p=dropout_rate)
    
    @overrides
    def get_output_dim(self) -> int:
        return self.base.get_output_dim()

    @overrides
    def forward(self, inputs):
        return self.dropout(self.base(inputs))