from src.inference import load_model, generate

def main():
	tokenizer, model, device = load_model()
	print("Local LLM Chat (type 'exit' to quit)")

	while True:
		user_input = input("\nUser: ")
		if user_input.lower() in ["exit", "quit"]:
			break

		reply = generate(tokenizer, model, device, user_input)
		print(f"Assistant: {reply}")

if __name__ == "__main__":
	main()
