from openai import OpenAI
client = OpenAI()

system_prompt = """
You are a Thai language evaluator, skilled in evaluate generated_text against the groundtruth.
Your task is to evaluate the generated_text by comparing it with the groundtruth associated with the query.
You answer should be a score between 0 and 1, where 0 means the generated_text is completely wrong and 1 means the generated_text is perfect.
Just response with a score without giving any reason.
"""
testcase = "/workspaces/llm_agents_experiment/data/testcases/TTB - testcase 2.csv"
# Load the testcases
import pandas as pd
# read the testcases
testcases = pd.read_csv(testcase)
# Dataframe to list of dictionaries
testcases = testcases.to_dict(orient="records")
# Create empty dataframe to store the results
results = pd.DataFrame(columns=["Query", "Groundtruth", "Generated text", "Score"])
for testcase in testcases:
    query = testcase["Query"]
    groundtruth = testcase["Expected answer"]
    generated_text = testcase["Agent answer"]
    user_prompt = f"query: {query}\ngroundtruth: {groundtruth}\ngenerated_text: {generated_text}"
    response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
        "role": "system",
        "content": system_prompt
        },
        {
        "role": "user",
        "content": user_prompt
        }
    ],
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    result = pd.DataFrame({"Query": query, "Groundtruth": groundtruth, "Generated text": generated_text, "Score": response.choices[0].message.content}, index=[0])
    results = pd.concat([results, result], ignore_index=True)
    print(response.choices[0].message.content)

# Save the results
results.to_csv("/workspaces/llm_agents_experiment/data/results/TTB - testcase 2 - GPT4-turbo - results.csv", index=False)