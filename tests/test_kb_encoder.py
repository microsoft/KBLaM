import json
from kblam.kb_encoder import KBEncoder

def test_kb_encoder():
    dataset = json.load(open('tests/sample_data.json'))
    all_keys = [row['name'] for row in dataset]
    all_values = [row['description'] for row in dataset]

    out_dim = 3072
    proj_kwargs = {'mlp_depth': 2,'mlp_hidden_dim': 512}

    kb_encoder = KBEncoder("all-MiniLM-L6-v2", 'mlp', out_dim, None, proj_kwargs, device='cpu')
    
    assert kb_encoder.encode_key(all_keys[0]).shape == kb_encoder.encode_val(all_values[0]).shape

    for k, v in kb_encoder.named_parameters():
        if v.requires_grad:
            assert ('projector' in k) or ('embedding' in k) 
