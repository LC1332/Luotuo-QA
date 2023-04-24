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
            question_answer = gr.Textbox(label="Answer", lines=2, interactive=False)
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
                                question_as = gr.Textbox(label="Question escapes to:", lines=2, interactive=False)
                            with gr.Row():
                                answer = gr.Textbox(label="Answer", lines=2, interactive=False)
                    with gr.Column():
                        # UI: Lora output^2
                        with gr.Box():
                            with gr.Row():
                                question_as_v2 = gr.Textbox(label="Question^2 escapes to:", lines=2, interactive=False)
                            with gr.Row():
                                answer_v2 = gr.Textbox(label="Answer^2:", lines=2, interactive=False)
        def inner_infer(story, question):
            question_answer = question_answer_infer(model, tokenizer, story, question)
            yield question_answer, "", "", "", "", ""
            for infer_out in infer_yield(model, tokenizer, story, question, origin_model = origin_model):
                yield question_answer, *infer_out
        submit.click(
                fn=inner_infer,
                inputs=[
                    story,
                    question,
                ],
                outputs=[
                    question_answer,
                    origin_answer,
                    question_as,
                    answer,
                    question_as_v2,
                    answer_v2,
                ]
            )
    b.queue().launch(prevent_thread_lock=True, share=share, server_port=server_port)
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)

    import time
    while True:
        time.sleep(0.5)