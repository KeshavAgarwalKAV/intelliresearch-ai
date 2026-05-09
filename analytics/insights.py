from langchain_groq import ChatGroq

def generate_ai_insights(summary, groq_api_key):

    prompt = f"""
You are a senior data scientist.

Analyze this dataset summary and provide insights.

DATASET SUMMARY:
{summary}

Provide:
1. Dataset overview
2. Potential data quality issues
3. Interesting observations
4. Suggested ML tasks
5. Recommended preprocessing steps
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

        print(f"Insight generation error: {e}")

        return None
    
def generate_correlation_insights(corr_matrix, groq_api_key):

    if corr_matrix is None:
        return None

    correlations = corr_matrix.to_string()

    prompt = f"""
You are a senior data scientist.

Analyze this correlation matrix.

CORRELATION MATRIX:
{correlations}

Provide:
1. Strong positive correlations
2. Strong negative correlations
3. Potential multicollinearity issues
4. Interesting feature relationships
5. ML implications
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

        print(f"Correlation insight error: {e}")

        return None