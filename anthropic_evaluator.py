import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

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
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=256,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    )
    result = pd.DataFrame({"Query": query, "Groundtruth": groundtruth, "Generated text": generated_text, "Score": response.content[0].text}, index=[0])
    results = pd.concat([results, result], ignore_index=True)
    print(response.content[0].text)

# Save the results
results.to_csv("/workspaces/llm_agents_experiment/data/results/TTB - testcase 2 - Claude3 Sonnet - results.csv", index=False)