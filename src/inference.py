from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(
	model_name: str = "microsoft/Phi-3-mini-4k-instruct",
	device: str | None = None,
):
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.unk_token
		tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
	tokenizer.padding_side = "right"

	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype=torch.float16 if device == "cuda" else torch.float32,
		device_map="auto" if device == "cuda" else None,
#		trust_remote_code=True,
	)

	model.config.pad_token_id = tokenizer.pad_token_id

	return tokenizer, model, device

def generate(
	tokenizer,
	model,
	device,
	user_message: str,
	max_new_tokens: int = 256,
	temperature: float = 0.7,
	top_p: float = 0.9,
):
	prompt = f"<|user|>\n{user_message}\n<|end|>\n<|assistant|>"

	inputs = tokenizer(
		prompt,
		return_tensors="pt",
		padding=True,
	)
	input_ids = inputs["input_ids"].to(device)
	attention_mask = inputs["attention_mask"].to(device)

	with torch.no_grad():
		output_ids = model.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			max_new_tokens=max_new_tokens,
			do_sample=True,
			temperature=temperature,
			top_p=top_p,
			eos_token_id=tokenizer.eos_token_id,
			pad_token_id=tokenizer.eos_token_id,
		)

	generated_ids = output_ids[0, input_ids.shape[-1]:]
	response = tokenizer.decode(generated_ids, skip_special_tokens=True)
	return response.strip()
