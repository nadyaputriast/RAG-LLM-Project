from groq import Groq
from apiKey import GROQ

client = Groq(api_key = GROQ)

def converse_with_llm(prompt):
	chat_completion = client.chat.completions.create(
		messages=[
			{
				# system message to set the context for the LLM
				"role": "system",
				"content": "You are a movie recommendation assistant."
			},
			{
				# user message containing the input prompt
				"role": "user",
				"content": prompt
			},
		],
		model="llama-3.3-70b-versatile",
		temperature=0.7,
		max_tokens=1024,
		top_p=1,
		stop=None,
		stream=False
	)
	
	# Return the content of the response
	return chat_completion.choices[0].message.content