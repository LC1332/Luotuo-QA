import gradio as gr

from infer import get_model, gen
from transformers import AutoTokenizer

example_story = """北京时间2月13日凌晨,2023年ATP250达拉斯站男单决赛。中国球员吴易昺先输一盘后挽救4个赛点并兑现第5个冠军点,最终以6(4)-7/7-6(3)/7-6(12)逆转惊险击败赛会5号种子、美国大炮伊斯内尔,就此改写历史,成为公开赛年代首位夺得ATP巡回赛男单冠军的中国大陆球员,并创造中国大陆球员的男单最高排名!

第一盘比赛,吴易昺在第12局错过了一个盘点,并最终抢七惜败;第二盘则挽救一个赛点后抢七局3-0领先开局,且以7-6(3)扳回一盘;第三盘决胜盘,在关键的第9局15-40落后情况下凭借连续的高质量发球逆转保发,之后比赛再次进入抢七,抢七局依然胶着,吴易昺又挽救了3个赛点,并兑现了自己的第5个冠军点,就此锁定冠军!历史性一刻到来时,吴易昺瞬间躺倒在地。全场比赛,伊斯内尔轰出了44记Ace球,但最终在主场依然输给了吴易昺。

凭借具有突破意义的这一冠,吴易昺在本周入账250个积分和112125美元的冠军奖金,在周一最新一期的男单排名榜单上,创中国大陆男网历史新高排名—第58位。根据比赛计划,吴易昺原本要出战本周进行的ATP250德拉海滩站,不过在达拉斯夺冠后,吴易昺因身体疲劳退出本站赛事,他的签位由幸运落败者约翰森替代。"""

example_question = "这场赛事中，谁是伊斯内尔的有力竞争者？"

def infer(model, tokenizer, story, question, origin_model = None):
    context = f"""给你下面的文本和问题，请先给出一个对应问题的同义转述，再给出问题的答案。
文本为：{story}
原始问题为：{question}
"""
    out = gen(model, tokenizer, context)
    question_as = out.split("答案为:")[0].split("问题转义为:")[1]
    answer = out.split("答案为:")[1]
    origin_out = ""
    if origin_model is not None:
        origin_out = gen(origin_model, tokenizer, context)
    print(f"### {context}: ###\n Origin: {origin_out}\n Lora: {out}")
    return question_as, answer, origin_out

def main(
    share: bool = False,
    server_port: int = None,
    model_name: str = "THUDM/chatglm-6b",
    peft_path: str = "./output",
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
                            origin_answer = gr.Textbox(label="ChatGLM-6B", lines=2, interactive=False)
                    with gr.Column():
                        # UI: Lora output
                        with gr.Row():
                            question_as = gr.Textbox(label="Question escapes to", lines=2, interactive=False)
                        with gr.Row():
                            answer = gr.Textbox(label="Answer", lines=2, interactive=False)
        def inner_infer(story, question):
            return infer(model, tokenizer, story, question, origin_model = origin_model)
        submit.click(
                fn=inner_infer,
                inputs=[
                    story,
                    question,
                ],
                outputs=[
                    question_as,
                    answer,
                    origin_answer,
                ],
                show_progress=True,
            )
    b.launch(prevent_thread_lock=True, share=share, server_port=server_port)
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)

    import time
    while True:
        time.sleep(0.5)