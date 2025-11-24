from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 텍스트
def text():
    text_model = SentenceTransformer("BAAI/bge-m3")
    text = "오늘의 날씨는?"
    text_emb = text_model.encode(text)
    return text_emb 

# 표
def table():
    table_str = "지역, 온도\n서울, 21도\n부산, 23도"
    table_emb = text_model.encode(table_str)
    return table_emb

# 이미지
def image():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    img = Image.open("sample.jpg")
    img_inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**img_inputs).detach().numpy().squeeze()
        return img_emb

# LLM
if __name__ == "__main__":

    text_emb = text()
    table_emb = table()
    image_emb = image()

    llm_ckpt = "meta-llama/Meta-Llama-3-8B"  # or "Qwen/Qwen1.5-14B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(llm_ckpt)
    model = AutoModelForCausalLM.from_pretrained(llm_ckpt, torch_dtype=torch.float16, device_map="auto")

    prompt = f"""아래 임베딩된 정보와 사용자 질문을 참고해서 답변해줘.

    - tes : {text_emb[:5]}...
    - table : {table_emb[:5]}...
    - image : {img_emb[:5]}...
    - 사용자 질문: '서울의 날씨와 관련 주요 특징은?'
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)
