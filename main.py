import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

MODEL = "poolside/laguna-xs-2.1"

tools = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Baca isi file. Panggil kalau user eksplisit minta lihat/baca/cek isi file.",                                                                                                                                 "parameters": {
                "type": "object",                                                                                                                                                                                                            "properties": {"filename": {"type": "string"}},
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Tulis/timpa isi file dengan konten baru yang LENGKAP. Panggil kalau user eksplisit minta ubah/tulis/upgrade/perbaiki file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "content"]
            }
        }
    }
]

def read(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def write(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"'{filename}' berhasil ditulis ({len(content)} karakter)"

def process_tool(name, args):
    if name == "read":
        return baca_file(args["filename"])
    elif name == "write":
        return tulis_file(args["filename"], args["content"])
    return f"Tool '{name}' tidak tersedia"

system_prompt = """Kamu adalah AI assistant yang bisa baca dan tulis (termasuk dirimu sendiri, app.py).

ATURAN:
- Ngobrol biasa / menyapa -> JANGAN panggil tool, jawab teks biasa.
- read -> hanya kalau user eksplisit minta baca/cek/lihat file.
- write -> hanya kalau user eksplisit minta ubah/tulis/upgrade/perbaiki file. Jelasin dulu rencana perubahan
  secara SINGKAT (maks 3-4 poin) dalam teks, LALU DI RESPONS YANG SAMA langsung panggil tool tulis
  dengan kode LENGKAP. JANGAN cuma bilang "nanti aku update" atau "ready" tanpa benar-benar memanggil tool.
- Sekali kamu bilang mau melakukan sesuatu, LANGSUNG lakukan panggil tool, jangan tunda dan jangan
  mengulang penjelasan yang sama di respons berikutnya.
- Bahasa: Indonesia."""

messages = []

while True:
    user_input = input("> ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    max_rounds = 4
    for i in range(max_rounds):
        force_text = i == max_rounds - 1  # ronde terakhir: paksa jawaban teks, biar gak nyangkut

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            tools=None if force_text else tools,
            temperature=0.1,
            max_tokens=4096,
        )

        msg = response.choices[0].message

        if not force_text and msg.tool_calls:
            # Simpan pesan assistant LENGKAP dengan tool_calls-nya (format resmi)
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    } for tc in msg.tool_calls
                ]
            })

            if msg.content:
                print(f"\n{msg.content}\n")

            # Eksekusi tiap tool call, hasilnya masuk sebagai role "tool" (bukan "user")
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                    result = process_tool(name, args)
                    print(f"[{name}] ✓")
                except Exception as e:
                    result = f"Gagal: {str(e)}"
                    print(f"[{name}] ({str(e)})")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
            continue

        # Jawaban teks final
        print(f"\n{msg.content}\n")
        messages.append({"role": "assistant", "content": msg.content})
        break
