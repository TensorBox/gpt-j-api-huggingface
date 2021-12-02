from transformers import AutoTokenizer, GPTJForCausalLM
import torch 

TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 1.0

class InferenceModel:
    def __init__(self):
        self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision='float16', 
                                                     torch_dtype=torch.half, low_cpu_mem_usage=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token # to avoid an error


    def run_batch_inference(self, params_json_list):
        inputs = self.tokenizer([params_json['prompt'] for params_json in params_json_list],
                                    return_tensors="pt", padding=True)

        # Currently, we cannot apply the following parameters for each item in a batch. 
        # Look at this issue: https://github.com/huggingface/transformers/issues/14530
        temperature = TEMPERATURE
        top_k = TOP_K
        top_p = TOP_P
        
        max_length = max([params_json["max_length"] for params_json in params_json_list])

        output_sequences = self.model.generate(input_ids=inputs['input_ids'].cuda(),
                                         attention_mask=inputs['attention_mask'].cuda(),
                                         do_sample=True, 
                                         temperature=temperature, 
                                         top_p=top_p, 
                                         top_k=top_k, 
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         max_length=max_length)
        outputs = [self.tokenizer.decode(x) for x in output_sequences]

        return outputs
