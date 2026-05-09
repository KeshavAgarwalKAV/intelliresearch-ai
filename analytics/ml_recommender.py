from langchain_groq import ChatGroq

def detect_target_candidates(df):

    candidates = []

    for col in df.columns:

        unique_values = df[col].nunique()

        # Likely classification target
        if unique_values <= 10:

            candidates.append({
                "column": col,
                "task": "classification",
                "unique_values": unique_values
            })

        # Likely regression target
        elif df[col].dtype != "object":

            candidates.append({
                "column": col,
                "task": "regression",
                "unique_values": unique_values
            })

    return candidates

def select_best_target(target_candidates):

    if not target_candidates:
        return None

    priority_keywords = [
        "target",
        "label",
        "class",
        "outcome",
        "y"
    ]

    # Prefer semantically meaningful targets
    for keyword in priority_keywords:

        for candidate in target_candidates:

            if keyword in candidate["column"].lower():

                return candidate

    # Otherwise prefer classification
    for candidate in target_candidates:

        if candidate["task"] == "classification":

            return candidate

    return target_candidates[0]

def generate_ml_recommendations(
    summary,
    target_candidates,
    groq_api_key
):

    prompt = f"""
You are a senior machine learning expert.

DATASET SUMMARY:
{summary}

TARGET CANDIDATES:
{target_candidates}

Analyze the dataset and provide:

1. Recommended ML tasks
2. Best target column candidates
3. Suggested preprocessing steps
4. Potential modeling approaches
5. Data quality concerns
6. Feature engineering ideas
"""

    try:

        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0.3
        )

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        print(f"ML recommendation error: {e}")

        return None
