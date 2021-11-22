from transformers import AutoTokenizer, GPTJForCausalLM
import torch 

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision='float16', torch_dtype=torch.half, low_cpu_mem_usage=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

def run_inference(params_json):
    input_ids = tokenizer(params_json['prompt'],
                        return_tensors="pt").input_ids.cuda()

    temperature = params_json["temperature"] if "temperature" in params_json else 1.0
    top_k = params_json["top_k"] if "top_k" in params_json else 50
    top_p = params_json["top_p"] if "top_p" in params_json else 1.0
    max_length = params_json["max_length"] if "max_length" in params_json else 20

    gen_tokens = model.generate(input_ids, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k, max_length=max_length)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return gen_text
