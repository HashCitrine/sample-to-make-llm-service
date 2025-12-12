FROM alpine:3.20 AS builder
RUN apk add --no-cache git cmake make gcc musl-dev python3 py3-pip
WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    pip install --no-cache-dir -r requirements.txt && \
    make LLAMA_CMAKE=1 -j$(nproc)

FROM alpine:3.20 AS runner
RUN apk add --no-cache python3
WORKDIR /app
COPY --from=builder /app/llama.cpp/llama.cpp .
COPY --from=builder /app/llama.cpp/convert-hf-to-gguf.py .
COPY --from=builder /app/llama.cpp/quantize .
ENTRYPOINT ["sh", "-c"]
CMD ["python3 convert-hf-to-gguf.py \"$MODEL_PATH\" --outfile /output/$OUTPUT_F16 --outtype f16 && ./quantize /output/$OUTPUT_F16 /output/$OUTPUT_QUANT q4_k_m"]
