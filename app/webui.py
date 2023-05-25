import gradio as gr

from infer import get_model, infer, infer_yield, example_story, example_question, question_answer_infer
from transformers import AutoTokenizer

def main(
    share: bool = False,
    server_port: int = None,
    model_name: str = "THUDM/chatglm-6b",
    peft_path: str = "silk-road/luotuo-qa-lora-0.1",
    model_revision: str = "969290547e761b20fdb96b0602b4fd8d863bbb85",
    with_origin_model: bool = True,
):
    model = get_model(model_name, peft_path)
    origin_model = None
    if with_origin_model:
        origin_model = get_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision = model_revision)

    with gr.Blocks() as b:
        with gr.Row():
            with gr.Row():
                with gr.Box():
                # UI: Story input
                    with gr.Row():
                        story = gr.TextArea(value=example_story, label="Story", lines=10, interactive=True)
                    with gr.Row():
                        with gr.Column(scale = 6):
                            # UI: Question input
                            question = gr.TextArea(value=example_question, label="Question", lines=2, interactive=True)
                        with gr.Column(scale = 1):
                            # UI: Submit button
                            submit = gr.Button("Ask", label="Submit", interactive=True, variant="primary")
        with gr.Row():
            with gr.Box():
                with gr.Row():
                    with gr.Column():
                        # UI: origin model output
                        with gr.Row():
                            origin_answer = gr.Textbox(label=model_name+":", lines=2, interactive=False)
                    with gr.Column():
                        # UI: Lora output
                        with gr.Box():
                            with gr.Row():
                                answer = gr.Textbox(label="Answer", lines=2, interactive=False)
        def inner_infer(story, question):
            for origin_out, answer in infer_yield(model, tokenizer, story, question, origin_model = origin_model):
                yield origin_out, answer
        submit.click(
                fn=inner_infer,
                inputs=[
                    story,
                    question,
                ],
                outputs=[
                    origin_answer,
                    answer,
                ]
            )
    b.queue().launch(prevent_thread_lock=True, share=share, server_port=server_port)
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)

    import time
    while True:
        time.sleep(0.5)