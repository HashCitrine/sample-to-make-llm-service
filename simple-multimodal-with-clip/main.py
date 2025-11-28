import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# 1. CLIP으로 멀티모달 임베딩 생성
clip_model_name = "openai/clip-vit-base-patch16"
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name).eval()

def get_multimodal_embedding(text, image_path):
    img = Image.open(image_path)
    inputs = clip_processor(text=[text], images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    return outputs.pooler_output.detach().numpy().squeeze()  # 512차원 임베딩

# 2. 벡터 저장소 초기화
dimension = 512
index = faiss.IndexFlatIP(dimension)  # 코사인 유사도

# 데이터 저장 (Memory)
data_embeddings = []
data_info = []

text1 = "서울의 맑은 날씨, 21도"
emb1 = get_multimodal_embedding(text1, "seoul_weather.jpg")
data_embeddings.append(emb1)
data_info.append({"text": text1, "image": "seoul_weather.jpg"})

text2 = "부산의 따뜻한 바다 날씨, 23도"
emb2 = get_multimodal_embedding(text2, "busan_weather.jpg")
data_embeddings.append(emb2)
data_info.append({"text": text2, "image": "busan_weather.jpg"})

# 정규화 후 FAISS에 추가
embeddings_np = np.array(data_embeddings).astype('float32')
faiss.normalize_L2(embeddings_np)
index.add(embeddings_np)

# 3. Load LLM
llm_ckpt = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_ckpt)
model = AutoModelForCausalLM.from_pretrained(llm_ckpt, torch_dtype=torch.float16, device_map="auto")

# 4. 쿼리 처리 및 RAG
def multimodal_rag_query(user_question, query_image_path):
    # 쿼리 임베딩 생성
    query_emb = get_multimodal_embedding(user_question, query_image_path)
    query_emb = query_emb.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_emb)
    
    # 유사도 검색 (Top-2)
    distances, indices = index.search(query_emb, 2)
    
    # 검색된 컨텍스트 수집
    context = []
    for idx in indices[0]:
        context.append(data_info[idx]["text"])
    
    # 프롬프트 생성
    prompt = f"""다음 컨텍스트를 참고하여 질문에 답변해줘:

    컨텍스트:
    {chr(10).join(context)}

    질문: {user_question}

    답변:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# example

if __name__ == "__main__":
    answer = multimodal_rag_query("서울의 날씨는 어떤가?", "query_weather.jpg")
    print("답변:", answer)
