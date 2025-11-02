import os
from google import genai  # Gemini SDK

MODEL_GEMINI = "gemini-2.5-flash"

client = None

def initialize_resources():
    print("Loading verifier resources...")
    global client
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY Not Found")
    client = genai.Client(api_key=gemini_api_key)
    print("Verifier resources loaded.")

def prompt_engineer(claim, evidences):
    
    # Build the prompt
    prompt_lines = []
    
    prompt_lines.append("""INSTRUCTIONS:
1. You are an expert fact-checker.
2. Carefully read the CLAIM and the provided EVIDENCES.
3. Evaluate the truthfulness of the CLAIM based on the provided EVIDENCES only.
4. Consider publication dates: more recent articles may override earlier contradictory evidence.
5. Classify the CLAIM into one of three categories:
   - "Real": if the CLAIM is well-supported by multiple credible EVIDENCES.
   - "Fake": if the CLAIM is contradicted by multiple credible EVIDENCES.
   - "Unverifiable": if there is insufficient or ambiguous EVIDENCE to support or refute the CLAIM.
6. Provide detailed reasoning but do not reference to the articles directly.
7. Return your final VERDICT in JSON format as shown below.
{
  "verdict": "Real" | "Fake" | "Unverifiable",
  "reasoning": "4-8 sentences explaining the rationale behind the verdict."
}
""")
    
    prompt_lines.append(f"CLAIM: {claim}\n")

    prompt_lines.append("EVIDENCES:\n")
    for idx, ev in enumerate(evidences, start=1):
        article = f"""Article {idx}:
- Published: {ev['published']}
- Categories: {', '.join(ev['categories'])}
- Entities: {', '.join(ev['entities'])}
- Title: {ev['title']}
- Content: {ev['content'][:200]}
"""
        prompt_lines.append(article)

    prompt_text = "\n".join(prompt_lines)
    return prompt_text

def verify_claim(claim, evidences):
    prompt = prompt_engineer(claim, evidences)
    
    # Call Gemini
    response = client.models.generate_content(
        model=MODEL_GEMINI,
        contents=prompt,
    )
    
    # Extract output
    verdict_text = response.text.strip()
    return verdict_text