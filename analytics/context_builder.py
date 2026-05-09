def build_dataset_context(df, summary):

    context = f"""
Dataset Overview

Rows: {summary['rows']}
Columns: {summary['columns']}

Column Names:
{', '.join(summary['column_names'])}

Numeric Columns:
{', '.join(summary['numeric_columns'])}

Categorical Columns:
{', '.join(summary['categorical_columns'])}

Missing Values:
"""

    for col, missing in summary["missing_values"].items():

        context += f"\n- {col}: {missing}"

    context += "\n\nData Types:\n"

    for col, dtype in summary["dtypes"].items():

        context += f"\n- {col}: {dtype}"

    context += "\n\nNumeric Statistics:\n"

    for col, stats in summary["numeric_summary"].items():

        context += f"\n{col}:\n"

        for stat_name, value in stats.items():

            context += f"  {stat_name}: {value}\n"

    return context