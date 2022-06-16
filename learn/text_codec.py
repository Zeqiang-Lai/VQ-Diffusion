from image_synthesis.utils.misc import instantiate_from_config


condition_codec_config = {
    "target": "image_synthesis.modeling.codecs.text_codec.tokenize.Tokenize",
    "params": {
        "context_length": 77,
        "add_start_and_end": True,
        "with_mask": True,
        "pad_value": 0,
        "clip_embedding": False,
        "tokenizer_config": {
            "target": "image_synthesis.modeling.modules.clip.simple_tokenizer.SimpleTokenizer",
            "params": {
                "end_idx": 49152
            }
        }
    }
}


condition_codec = instantiate_from_config(condition_codec_config)

import torch
device = torch.device('cuda')

@torch.no_grad()
def prepare_condition(batch, condition=None):
    cond_key = 'text'
    cond = batch[cond_key] if condition is None else condition
    if torch.is_tensor(cond):
        cond = cond.to(device)
    cond = condition_codec.get_tokens(cond)
    cond_ = {}
    for k, v in cond.items():
        v = v.to(device) if torch.is_tensor(v) else v
        cond_['condition_' + k] = v
    return cond_

from image_synthesis.data.cub200_dataset import Cub200Dataset

config = {
    "target": "image_synthesis.data.utils.image_preprocessor.DalleTransformerPreprocessor",
    "params": {
              "size": 256,
              "phase": "val"
    }

}
dataset = Cub200Dataset('../data/cub', phase='test', im_preprocessor_config=config)
loader = dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

batch = iter(loader).next()

cond = prepare_condition(batch)
print(cond)
# condition_token, condition_mask
#
#  {'condition_token': tensor([[49406,   589,  3329,   533,  1395,  2866,   267,   791,   320,  1538,
#          18138,  2886,   267,   537,  1606, 12464,   269, 49407,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0],
#  'condition_mask': tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
#           True,  True,  True,  True,  True,  True,  True,  True, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False, False, False, False,
#          False, False, False, False, False, False, False],