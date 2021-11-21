from transformers import AutoTokenizer, GPTJForCausalLM
import torch 

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision='float16', torch_dtype=torch.half, low_cpu_mem_usage=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

def run_inference(params_json):
    input_ids = tokenizer(params_json['prompt'],
                        return_tensors="pt").input_ids.cuda()

    gen_tokens = model.generate(input_ids, do_sample=True)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return gen_text
