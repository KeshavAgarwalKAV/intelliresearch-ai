import gradio as gr


def create_visualization_tabs():

    with gr.Tabs():

        with gr.Tab("📊 Insights"):

            process_status = gr.Markdown(
                value="*Upload files and click Process to begin*"
            )

        with gr.Tab("📈 Visualizations"):

            missing_plot = gr.Plot(
                label="📉 Missing Value Analysis"
            )

            correlation_plot = gr.Plot(
                label="📈 Correlation Heatmap"
            )

            feature_plot = gr.Plot(
                label="⭐ Feature Importance"
            )

    return (
        process_status,
        missing_plot,
        correlation_plot,
        feature_plot
    )